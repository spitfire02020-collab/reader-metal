import Foundation
import Metal

// MARK: - MetalLMBackend

/// Core Metal kernel dispatcher for GPT-2-style language model forward pass.
/// Handles per-layer computation using custom Metal kernels (no MPS/ORT dependency).
///
/// Architecture per layer:
///   hidden → preLN → QKV_proj → RoPE → Attention → O_proj → residual_add → postLN → FFN → residual_add → hidden
public final class MetalLMBackend: LanguageModelBackend {

    // MARK: - Metal Objects

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var library: MTLLibrary!

    // Pipeline states
    private var gemmNTPipeline: MTLComputePipelineState!
    private var gemmNNPipeline: MTLComputePipelineState!
    private var tanhGeluPipeline: MTLComputePipelineState!
    private var attentionPipeline: MTLComputePipelineState!
    private var layerNormPipeline: MTLComputePipelineState!
    private var residualAddPipeline: MTLComputePipelineState!
    private var ropeKernelPipeline: MTLComputePipelineState!

    // KV Cache
    private var kvCache: KVCacheManager!

    // Weight buffers (float16) — keyed by ONNX tensor name from weights_manifest.json
    private var weightBuffers: [String: MTLBuffer] = [:]

    // Weights directory (set during init)
    private let weightsDir: URL

    // Pre-allocated activation buffers (half/float16)
    // Shapes: row-major, [rows, cols] or flattened as [total]
    // NOTE: For GPT2NoEmbed, QKV = 3072 = 16*64*3 (standard MHA, not GQA)
    private var qkvBuffer: MTLBuffer!           // [1, 1, 3072] = 3072 half (GPT2NoEmbed QKV fused output)
    private var qBuffer: MTLBuffer!               // [16, 64] = 1024 half (GPT2NoEmbed Q)
    private var kWriteBuffer: MTLBuffer!          // [16, 64] = 1024 half (GPT2NoEmbed K)
    private var vWriteBuffer: MTLBuffer!          // [16, 64] = 1024 half (GPT2NoEmbed V)
    private var kBuffer: MTLBuffer!              // [16, maxSeqLen, 64] half (KV cache key buffer)
    private var vBuffer: MTLBuffer!              // [16, maxSeqLen, 64] half (KV cache value buffer)
    private var attnOutBuffer: MTLBuffer!        // [16, 64] = 1024 half (GPT2NoEmbed attn output)
    private var ln1OutBuffer: MTLBuffer!         // [1, 1, 1024] = 1024 half
    private var residualBuffer: MTLBuffer!        // [1, 1, 1024] = 1024 half
    private var ln2OutBuffer: MTLBuffer!         // [1, 1, 1024] = 1024 half
    private var ffnInterimBuffer: MTLBuffer!    // [1, 1, 4096] = 4096 half
    private var ffnOutBuffer: MTLBuffer!         // [1, 1, 1024] = 1024 half
    private var logitsBuffer: MTLBuffer!         // [vocabSize] float32 (final output)
    private var finalLnOutBuffer: MTLBuffer!    // [1, 1, 1024] = 1024 half

    // RoPE LUT buffers (float32)
    private var ropeCosBuffer: MTLBuffer!        // [maxSeqLen, 32] float32
    private var ropeSinBuffer: MTLBuffer!       // [maxSeqLen, 32] float32

    // Causal mask buffer (float32, [maxSeqLen, maxSeqLen])
    private var causalMaskBuffer: MTLBuffer!

    // Config (local copies for convenience)
    private let numLayers = MetalLMConfig.numLayers
    private let numQHeads = MetalLMConfig.numQueryHeads
    private let numKVHeads = MetalLMConfig.numKVHeads
    private let headDim = MetalLMConfig.headDim
    private let hiddenSize = MetalLMConfig.hiddenSize
    private let intermediateSize = MetalLMConfig.intermediateSize
    private let vocabSize = MetalLMConfig.vocabSize
    private let maxSeqLen = MetalLMConfig.maxSequenceLength
    private let ropeDimHalf: Int  // headDim / 2 = 32

    // GPT2NoEmbed uses standard MHA: 16 heads × 64 dim = 1024 per Q/K/V
    // This differs from MetalLMBackend's GQA design (80 Q + 16 KV).
    // Buffer sizes here use 16 for Q/K/V to match GPT2NoEmbed.
    private let gpt2NumHeads = 16

    // MARK: - Initialization

    public init(device: MTLDevice, weightsDir: URL) throws {
        NSLog("[MetalLMBackend] init START, device=%@", device.name)
        self.device = device
        self.weightsDir = weightsDir
        guard let q = device.makeCommandQueue() else {
            NSLog("[MetalLMBackend] init FAIL: makeCommandQueue returned nil")
            throw MetalLMError.commandQueueFailed
        }
        self.commandQueue = q
        self.ropeDimHalf = headDim / 2
        NSLog("[MetalLMBackend] init: commandQueue OK, loading weights from %@", weightsDir.path)

        // Load weights from manifest immediately after device/queue setup
        // This ensures weightBuffers are populated before any forward pass
        try loadWeights()
        NSLog("[MetalLMBackend] init: weights loaded OK")
    }

    // MARK: - LanguageModelBackend

    public func initialize(
        numLayers: Int,
        numKVHeads: Int,
        headDim: Int,
        maxSeqLen: Int,
        device: MTLDevice
    ) async throws {
        NSLog("[MetalLMBackend] initialize START")
        try compilePipelines()
        NSLog("[MetalLMBackend] initialize: pipelines compiled OK")
        // Note: loadWeights() was already called in init via weightsDir
        allocateBuffers()
        NSLog("[MetalLMBackend] initialize: buffers allocated OK")
        try precomputeRoPE()
        NSLog("[MetalLMBackend] initialize: RoPE precomputed OK")
        try precomputeCausalMask()
        NSLog("[MetalLMBackend] initialize: causal mask OK")

        // KVCacheManager must be created AFTER allocateBuffers so kBuffer/vBuffer are ready
        self.kvCache = KVCacheManager(
            numLayers: self.numLayers,
            numKVHeads: self.numKVHeads,
            headDim: self.headDim,
            maxSeqLen: self.maxSeqLen,
            device: self.device
        )
        // Wire up KV cache backing buffers (per-layer; scratch space is layer 0's buffers)
        // Note: kBuffer/vBuffer here are scratch/placeholder.
        // The actual per-layer K/V cache is accessed via kvCache.buffer(layer, isKey) in each forward call.
        self.kBuffer = kvCache.buffer(for: 0, isKey: true)
        self.vBuffer = kvCache.buffer(for: 0, isKey: false)
        // This is a simplification — the real implementation uses kvCache.buffer(layer, isKey).
    }

    public func forward(
        inputsEmbds: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        var hidden = inputsEmbds

        for layer in 0..<numLayers {
            hidden = try forwardLayer(
                layer: layer,
                hidden: hidden,
                kvWriteOffset: kvWriteOffset,
                kvReadLength: kvReadLength,
                commandBuffer: commandBuffer
            )
        }

        return try finalForward(hidden: hidden, commandBuffer: commandBuffer)
    }

    public func reset() async {
        await kvCache.reset()
    }

    // MARK: - Per-Layer Forward

    /// Runs one transformer layer's computation.
    /// hidden: [1, 1, 1024] half input
    /// Returns hidden state [1, 1, 1024] half for the next layer
    public func forwardLayer(
        layer: Int,
        hidden: MTLBuffer,
        kvWriteOffset: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        // 1. Pre-LayerNorm: ln1_out = LayerNorm(hidden, ln1_w, ln1_b)
        try runLayerNorm(
            input: hidden,
            gamma: weight("h.\(layer).ln_1.weight"),
            beta: weight("h.\(layer).ln_1.bias"),
            output: ln1OutBuffer,
            dim: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 2. QKV projection: qkv = gemm_nt(ln1_out, c_attn.weight)
        // ln1_out: [1,1,1024], c_attn.weight: [1024, 3072] in ONNX → [3072, 1024] in memory
        // gemm_nt: C = A @ B^T, A[M,K], B[N,K] row-major → C[M,N]
        // M=1, N=3072 (16*64*3 for standard MHA), K=1024
        try runGEMM_NT(
            A: ln1OutBuffer,
            B: weight("h.\(layer).attn.c_attn.weight"),
            C: qkvBuffer,
            M: 1,
            N: gpt2NumHeads * headDim * 3,  // 3072 = 16*64*3 (Q+K+V for standard MHA)
            K: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 3. Unpack Q/K/V from qkvBuffer [1, 1, 3072]
        // GPT2NoEmbed standard MHA: Q/K/V each 16 heads × 64 dim = 1024 elements
        // Q: [0..1024) → qBuffer[16,64]
        // K: [1024..2048) → kWriteBuffer[16,64]
        // V: [2048..3072) → vWriteBuffer[16,64]
        unpackQKV()

        // 4. Apply RoPE to Q and K at current position
        try applyRoPELayer(
            layer: layer,
            q: qBuffer,
            k: kWriteBuffer,
            seqPos: kvWriteOffset,
            commandBuffer: commandBuffer
        )

        // 5. Write K/V to KV cache at kvWriteOffset for this layer
        writeKVPairs(layer: layer, kvWriteOffset: kvWriteOffset)

        // 6. Attention: attn_out = attention_decode_step(q, k_cache, v_cache, kvReadLength)
        try runAttention(
            q: qBuffer,
            layer: layer,
            kvReadLength: kvReadLength,
            commandBuffer: commandBuffer
        )

        // 7. O projection: o = gemm_nt(attn_out, c_proj.weight)
        // attn_out: [1,1,1024], c_proj.weight: [1024, 1024] in ONNX → [1024, 1024] in memory
        // M=1, N=1024, K=1024
        try runGEMM_NT(
            A: attnOutBuffer,
            B: weight("h.\(layer).attn.c_proj.weight"),
            C: residualBuffer,
            M: 1,
            N: hiddenSize,
            K: gpt2NumHeads * headDim,  // 1024 for GPT2NoEmbed
            commandBuffer: commandBuffer
        )

        // 8. Residual add: residual = hidden + o → stored in residualBuffer
        try runResidualAdd(
            a: hidden,
            b: residualBuffer,
            output: residualBuffer,
            size: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 9. Post-LayerNorm: ln2_out = LayerNorm(residual, ln2_w, ln2_b)
        try runLayerNorm(
            input: residualBuffer,
            gamma: weight("h.\(layer).ln_2.weight"),
            beta: weight("h.\(layer).ln_2.bias"),
            output: ln2OutBuffer,
            dim: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 10. FFN gate: ffn_interim = gelu(gemm_nt(ln2_out, c_fc.weight))
        // ln2_out: [1,1,1024], c_fc.weight: [1024, 4096] in ONNX → [4096, 1024] in memory
        // M=1, N=4096, K=1024
        try runGEMM_NT(
            A: ln2OutBuffer,
            B: weight("h.\(layer).mlp.c_fc.weight"),
            C: ffnInterimBuffer,
            M: 1,
            N: intermediateSize,
            K: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 11. GELU activation (in-place on ffnInterimBuffer)
        try runTanhGelu(
            input: ffnInterimBuffer,
            output: ffnInterimBuffer,
            size: intermediateSize,
            commandBuffer: commandBuffer
        )

        // 12. FFN up: ffn_hidden = gemm_nt(ffn_interim, c_proj.weight)
        // ffn_interim: [1,1,4096], c_proj.weight: [4096, 1024] in ONNX → [1024, 4096] in memory
        // M=1, N=1024, K=4096 (gemm_nt: C = A @ B^T where B[N,K]=[1024,4096])
        try runGEMM_NT(
            A: ffnInterimBuffer,
            B: weight("h.\(layer).mlp.c_proj.weight"),
            C: ffnOutBuffer,
            M: 1,
            N: hiddenSize,
            K: intermediateSize,
            commandBuffer: commandBuffer
        )

        // 13. Residual add: hidden = residual + ffn_hidden → residualBuffer (in-place)
        try runResidualAdd(
            a: residualBuffer,
            b: ffnOutBuffer,
            output: residualBuffer,
            size: hiddenSize,
            commandBuffer: commandBuffer
        )

        return residualBuffer
    }

    /// Final forward: LayerNorm + LM head → logits
    /// hidden: [1, 1, 1024] half → returns logits [vocabSize] float32
    public func finalForward(
        hidden: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        // Final LayerNorm
        try runLayerNorm(
            input: hidden,
            gamma: weight("ln_f.weight"),
            beta: weight("ln_f.bias"),
            output: finalLnOutBuffer,
            dim: hiddenSize,
            commandBuffer: commandBuffer
        )

        // LM head: logits = gemm_nt(ln_f_out, lm_head.weight)
        // ln_f_out: [1,1,1024], lm_head.weight: [6563, 1024] in ONNX → [1024, 6563] in memory
        // gemm_nt: C = A @ B^T where B[N,K]=[6563,1024] → B^T[K,N]=[1024,6563]
        // M=1, N=vocabSize=6563, K=1024 → result [1, 6563]
        try runGEMM_NT(
            A: finalLnOutBuffer,
            B: weight("lm_head.weight"),
            C: logitsBuffer,
            M: 1,
            N: vocabSize,
            K: hiddenSize,
            commandBuffer: commandBuffer
        )

        return logitsBuffer
    }

    // MARK: - Kernel Dispatch Helpers

    /// GEMM with B transposed: C = A @ B^T
    /// A: [M,K], B: [N,K] row-major → C: [M,N]
    private func runGEMM_NT(
        A: MTLBuffer,
        B: MTLBuffer,
        C: MTLBuffer,
        M: Int,
        N: Int,
        K: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(gemmNTPipeline)
        enc.setBuffer(A, offset: 0, index: 0)
        enc.setBuffer(B, offset: 0, index: 1)
        enc.setBuffer(C, offset: 0, index: 2)

        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)

        let tgW = 16, tgH = 16
        let threadsPerGroup = MTLSize(width: tgW, height: tgH, depth: 1)
        let numThreadGroups = MTLSize(
            width: (N + tgW - 1) / tgW,
            height: (M + tgH - 1) / tgH,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    /// Standard GEMM: C = A @ B
    /// A: [M,K], B: [K,N] row-major → C: [M,N]
    private func runGEMM_NN(
        A: MTLBuffer,
        B: MTLBuffer,
        C: MTLBuffer,
        M: Int,
        N: Int,
        K: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(gemmNNPipeline)
        enc.setBuffer(A, offset: 0, index: 0)
        enc.setBuffer(B, offset: 0, index: 1)
        enc.setBuffer(C, offset: 0, index: 2)

        var m = UInt32(M), n = UInt32(N), k = UInt32(K)
        enc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)

        let tgW = 16, tgH = 16
        let threadsPerGroup = MTLSize(width: tgW, height: tgH, depth: 1)
        let numThreadGroups = MTLSize(
            width: (N + tgW - 1) / tgW,
            height: (M + tgH - 1) / tgH,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    /// LayerNorm: output = (input - mean) / sqrt(var + eps) * gamma + beta
    /// Dispatch: threads=256, threadgroups=1 (layer_norm kernel expects grid.y=batch)
    private func runLayerNorm(
        input: MTLBuffer,
        gamma: MTLBuffer,
        beta: MTLBuffer,
        output: MTLBuffer,
        dim: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(layerNormPipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(beta, offset: 0, index: 2)
        enc.setBuffer(output, offset: 0, index: 3)

        var d = UInt32(dim)
        var eps: Float16 = Float16(1e-5)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&eps, length: MemoryLayout<Float16>.size, index: 5)

        // layer_norm: grid.y = batch_size. We use batch=1
        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: 1, height: 1, depth: 1)
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    /// TanhGelu activation (element-wise, works in-place or copy)
    private func runTanhGelu(
        input: MTLBuffer,
        output: MTLBuffer,
        size: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(tanhGeluPipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(output, offset: 0, index: 1)

        var s = UInt32(size)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 2)

        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: (size + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    /// Attention decode step (single position Q with per-layer KV cache).
    /// attention_decode_step kernel: Grid: (80, 64), Threadgroup: (16, 4)
    /// NOTE: MetalLMBackend attention kernel is designed for GQA (80 Q + 16 KV, ratio=5).
    /// GPT2NoEmbed uses standard MHA (16 Q + 16 KV, ratio=1). The kernel's
    /// Q_KV_RATIO=5 mapping (q_head / 5) is incorrect for standard MHA.
    /// Weight loading is correct; attention kernel needs separate MHA adaptation.
    private func runAttention(
        q: MTLBuffer,
        layer: Int,
        kvReadLength: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        let kBuf = kvCache.buffer(for: layer, isKey: true)
        let vBuf = kvCache.buffer(for: layer, isKey: false)

        enc.setComputePipelineState(attentionPipeline)
        enc.setBuffer(q, offset: 0, index: 0)          // Q [80, 64]
        enc.setBuffer(kBuf, offset: 0, index: 1)      // K [16, maxSeq, 64]
        enc.setBuffer(vBuf, offset: 0, index: 2)     // V [16, maxSeq, 64]
        enc.setBuffer(attnOutBuffer, offset: 0, index: 3)  // output [80, 64]

        var kvLen = UInt32(kvReadLength)
        var maxSeq = UInt32(maxSeqLen)
        enc.setBytes(&kvLen, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&maxSeq, length: MemoryLayout<UInt32>.size, index: 5)

        // Grid: MTLSize(width: 80, height: 64, depth: 1)
        // Threadgroup: MTLSize(width: 16, height: 4, depth: 1)
        let threadsPerGroup = MTLSize(width: 16, height: 4, depth: 1)
        let numThreadGroups = MTLSize(width: 80, height: 64, depth: 1)
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    /// Residual add: out = a + b (element-wise)
    /// Dispatch: 1D grid, threads=256
    private func runResidualAdd(
        a: MTLBuffer,
        b: MTLBuffer,
        output: MTLBuffer,
        size: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(residualAddPipeline)
        enc.setBuffer(a, offset: 0, index: 0)
        enc.setBuffer(b, offset: 0, index: 1)
        enc.setBuffer(output, offset: 0, index: 2)

        var s = UInt32(size)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)

        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: (size + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    // MARK: - RoPE

    /// Apply RoPE to Q and K for a specific layer and sequence position.
    /// Q: [80, 64] half (modified in-place after kernel writes back)
    /// K: [16, 64] half write buffer (written to kBuffer after RoPE)
    /// seqPos: the sequence position for LUT lookup
    private func applyRoPELayer(
        layer: Int,
        q: MTLBuffer,
        k: MTLBuffer,
        seqPos: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(ropeKernelPipeline)
        enc.setBuffer(q, offset: 0, index: 0)
        enc.setBuffer(k, offset: 0, index: 1)
        enc.setBuffer(ropeCosBuffer, offset: 0, index: 2)
        enc.setBuffer(ropeSinBuffer, offset: 0, index: 3)

        var position = UInt32(seqPos)
        var maxSeqU = UInt32(maxSeqLen)
        var numQ = UInt32(numQHeads)
        var numKV = UInt32(numKVHeads)
        var hd = UInt32(headDim)
        enc.setBytes(&position, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&maxSeqU, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&numQ, length: MemoryLayout<UInt32>.size, index: 6)
        enc.setBytes(&numKV, length: MemoryLayout<UInt32>.size, index: 7)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 8)

        // Grid: (num_heads_total, head_dim/2) = (96, 32)
        let threadsPerGroup = MTLSize(width: 16, height: 4, depth: 1)
        let numThreadGroups = MTLSize(
            width: (numQHeads + numKVHeads + 15) / 16,
            height: (ropeDimHalf + 3) / 4,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    // MARK: - QKV Unpacking

    /// Unpack fused qkvBuffer [1, 1, 3072] into Q, K, V write buffers.
    /// GPT2NoEmbed standard MHA: Q/K/V each 16 heads × 64 dim = 1024 elements
    /// Q: qBuffer [16, 64]
    /// K: kWriteBuffer [16, 64]
    /// V: vWriteBuffer [16, 64]
    private func unpackQKV() {
        // GPT2NoEmbed: qkv = [1, 1, 3072] = 16*64*3 = Q + K + V
        // Layout: Q[0..1024), K[1024..2048), V[2048..3072)
        let qkvPtr = qkvBuffer.contents().bindMemory(to: Float16.self, capacity: 3072)
        let qPtr = qBuffer.contents().bindMemory(to: Float16.self, capacity: 1024)
        let kPtr = kWriteBuffer.contents().bindMemory(to: Float16.self, capacity: 1024)
        let vPtr = vWriteBuffer.contents().bindMemory(to: Float16.self, capacity: 1024)

        let qBytes = 1024 * MemoryLayout<Float16>.size
        let kBytes = 1024 * MemoryLayout<Float16>.size

        memcpy(qPtr, qkvPtr, qBytes)
        memcpy(kPtr, qkvPtr.advanced(by: 1024), kBytes)
        memcpy(vPtr, qkvPtr.advanced(by: 2048), kBytes)
    }

    /// Write K/V write buffers into the KV cache for a specific layer and position.
    private func writeKVPairs(layer: Int, kvWriteOffset: Int) {
        let kStride = numKVHeads * headDim  // 16 * 64 = 1024 elements
        let byteOffset = kvWriteOffset * kStride * MemoryLayout<Float16>.size

        let layerKeyBuf = kvCache.buffer(for: layer, isKey: true)
        let layerValBuf = kvCache.buffer(for: layer, isKey: false)

        memcpy(layerKeyBuf.contents().advanced(by: byteOffset),
               kWriteBuffer.contents(),
               kStride * MemoryLayout<Float16>.size)
        memcpy(layerValBuf.contents().advanced(by: byteOffset),
               vWriteBuffer.contents(),
               kStride * MemoryLayout<Float16>.size)
    }

    // MARK: - Weight Manifest Entry

    private struct WeightManifestEntry: Codable {
        let fp16: String?
    }

    // MARK: - Initialization Helpers

    private func compilePipelines() throws {
        // Try to load the default library (from compiled .metal sources)
        // Falls back to loading default.metallib from the bundle as a resource file.
        var lib: MTLLibrary?
        if let defaultLib = device.makeDefaultLibrary() {
            lib = defaultLib
        } else if let metallibURL = Bundle.main.url(forResource: "default", withExtension: "metallib") {
            lib = try? device.makeLibrary(URL: metallibURL)
        }
        guard let library = lib else {
            throw MetalLMError.kernelNotFound("Metal library not found. Ensure .metal files are compiled or default.metallib is in the bundle.")
        }
        self.library = library

        func makePipeline(_ name: String) throws -> MTLComputePipelineState {
            guard let f = library.makeFunction(name: name) else {
                throw MetalLMError.kernelNotFound(name)
            }
            return try device.makeComputePipelineState(function: f)
        }

        gemmNTPipeline = try makePipeline("gemm_nt")
        gemmNNPipeline = try makePipeline("gemm_nn")
        tanhGeluPipeline = try makePipeline("tanh_gelu_kernel")
        attentionPipeline = try makePipeline("attention_decode_step")
        layerNormPipeline = try makePipeline("layer_norm")
        residualAddPipeline = try makePipeline("residual_add")
        ropeKernelPipeline = try makePipeline("rope_apply_kernel")
    }

    private func allocateBuffers() {
        let fp16 = MemoryLayout<Float16>.size
        let fp32 = MemoryLayout<Float32>.size

        // GPT2NoEmbed buffer sizes (standard MHA: 16 heads × 64 dim per Q/K/V)
        // qkvBuffer: [1, 1, 3072] = 16*64*3 = Q+K+V for standard MHA
        qkvBuffer          = device.makeBuffer(length: 1 * 1 * gpt2NumHeads * headDim * 3 * fp16, options: .storageModeShared)!
        // qBuffer: [16, 64] = 1024 half (GPT2NoEmbed Q)
        qBuffer            = device.makeBuffer(length: gpt2NumHeads * headDim * fp16, options: .storageModeShared)!
        // kWriteBuffer/vWriteBuffer: [16, 64] = 1024 half each (GPT2NoEmbed K/V)
        kWriteBuffer       = device.makeBuffer(length: gpt2NumHeads * headDim * fp16, options: .storageModeShared)!
        vWriteBuffer       = device.makeBuffer(length: gpt2NumHeads * headDim * fp16, options: .storageModeShared)!
        // kBuffer/vBuffer: allocated after KVCacheManager creation; placeholder here
        // attnOutBuffer: [16, 64] = 1024 half (GPT2NoEmbed attention output)
        attnOutBuffer      = device.makeBuffer(length: gpt2NumHeads * headDim * fp16, options: .storageModeShared)!
        ln1OutBuffer       = device.makeBuffer(length: hiddenSize * fp16, options: .storageModeShared)!
        residualBuffer     = device.makeBuffer(length: hiddenSize * fp16, options: .storageModeShared)!
        ln2OutBuffer       = device.makeBuffer(length: hiddenSize * fp16, options: .storageModeShared)!
        ffnInterimBuffer   = device.makeBuffer(length: intermediateSize * fp16, options: .storageModeShared)!
        ffnOutBuffer       = device.makeBuffer(length: hiddenSize * fp16, options: .storageModeShared)!
        logitsBuffer       = device.makeBuffer(length: vocabSize * fp32, options: .storageModeShared)!
        finalLnOutBuffer   = device.makeBuffer(length: hiddenSize * fp16, options: .storageModeShared)!

        // RoPE LUT: [maxSeqLen, headDim/2] = [1500, 32] float32
        let ropeLutSize = maxSeqLen * ropeDimHalf * fp32
        ropeCosBuffer  = device.makeBuffer(length: ropeLutSize, options: .storageModeShared)!
        ropeSinBuffer  = device.makeBuffer(length: ropeLutSize, options: .storageModeShared)!

        // Causal mask: [maxSeqLen, maxSeqLen] float32 (for prefill; reserved for decode)
        let causalSize = maxSeqLen * maxSeqLen * fp32
        causalMaskBuffer = device.makeBuffer(length: causalSize, options: .storageModeShared)!
    }

    private func precomputeRoPE() throws {
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        guard let ropeLutFunc = library.makeFunction(name: "compute_rope_lut") else {
            throw MetalLMError.kernelNotFound("compute_rope_lut")
        }
        let ropeLutPipeline = try device.makeComputePipelineState(function: ropeLutFunc)

        enc.setComputePipelineState(ropeLutPipeline)
        enc.setBuffer(ropeCosBuffer, offset: 0, index: 0)
        enc.setBuffer(ropeSinBuffer, offset: 0, index: 1)

        var maxSeqU = UInt32(maxSeqLen)
        var headDimU = UInt32(headDim)
        enc.setBytes(&maxSeqU, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&headDimU, length: MemoryLayout<UInt32>.size, index: 3)

        let threadsPerGroup = MTLSize(width: 16, height: 4, depth: 1)
        let numThreadGroups = MTLSize(
            width: (maxSeqLen + 15) / 16,
            height: (ropeDimHalf + 3) / 4,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)

        cmd.commit()
        cmd.waitUntilCompleted()
    }

    private func precomputeCausalMask() throws {
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        guard let maskFunc = library.makeFunction(name: "causal_mask_kernel") else {
            throw MetalLMError.kernelNotFound("causal_mask_kernel")
        }
        let maskPipeline = try device.makeComputePipelineState(function: maskFunc)

        enc.setComputePipelineState(maskPipeline)
        enc.setBuffer(causalMaskBuffer, offset: 0, index: 0)

        var seqLen = UInt32(maxSeqLen)
        enc.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 1)

        let totalThreads = maxSeqLen * maxSeqLen
        let threadsPerGroup = MTLSize(width: 1024, height: 1, depth: 1)
        let numThreadGroups = MTLSize(
            width: (totalThreads + 1023) / 1024,
            height: 1, depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)

        cmd.commit()
        cmd.waitUntilCompleted()
    }

    private func loadWeights() throws {
        // Load weights from weights_manifest.json
        // GPT2NoEmbed uses standard MHA (16 heads × 64 = 1024 per Q/K/V)
        // with matmul convention: ONNX stores [out_dim, in_dim], gemm_nt is C = A @ B^T
        let manifestPath = weightsDir.appendingPathComponent("weights_manifest.json")
        let manifestData = try Data(contentsOf: manifestPath)
        let manifest = try JSONDecoder().decode([String: WeightManifestEntry].self, from: manifestData)

        for (name, entry) in manifest {
            guard let fp16File = entry.fp16 else { continue }
            let fp16Data = try Data(contentsOf: weightsDir.appendingPathComponent(fp16File))
            let fp16Buf = fp16Data.withUnsafeBytes { ptr in
                device.makeBuffer(bytes: ptr.baseAddress!, length: fp16Data.count, options: .storageModeShared)!
            }
            weightBuffers[name] = fp16Buf
        }

        // Note: c_attn.bias [3072] and c_proj.bias [1024] are loaded into weightBuffers
        // but not yet applied in forwardLayer. Bias application is a future enhancement.
    }

    // MARK: - Weight Access

    private func weight(_ name: String) throws -> MTLBuffer {
        guard let buf = weightBuffers[name] else {
            throw MetalLMError.weightNotLoaded(name)
        }
        return buf
    }
}

