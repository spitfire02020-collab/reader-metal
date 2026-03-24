import Foundation
@preconcurrency import Metal

// MARK: - MetalLMBackend

/// Core Metal kernel dispatcher for GPT-2-style language model forward pass.
/// Handles per-layer computation using custom Metal kernels (no MPS/ORT dependency).
///
/// Architecture per layer:
///   hidden → preLN → QKV_proj → RoPE → Attention → O_proj → residual_add → postLN → FFN → residual_add → hidden
public final class MetalLMBackend: LanguageModelBackend, @unchecked Sendable {

    // MARK: - Metal Objects

    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private var library: MTLLibrary!

    // Pipeline states
    private var gemmNTPipeline: MTLComputePipelineState!
    private var gemmNNPipeline: MTLComputePipelineState!
    private var tanhGeluPipeline: MTLComputePipelineState!
    private var attentionPipeline: MTLComputePipelineState!
    private var prefillAttentionPipeline: MTLComputePipelineState!
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

    // Prefill attention buffers (full sequence per forward pass)
    // [1, numHeads, maxSeqLen, headDim] half ≈ 3 MB per buffer
    private var prefillQBuffer: MTLBuffer!
    private var prefillKBuffer: MTLBuffer!
    private var prefillVBuffer: MTLBuffer!
    // [1, numHeads, maxSeqLen, headDim] float32 output
    private var prefillAttnOutBuffer: MTLBuffer!

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

        // DIAGNOSTIC TEST: Run LayerNorm in isolation with known input
        do {
            NSLog("[MetalLMBackend] TEST: Running LayerNorm isolation test...")
            // Fill input buffer with known values: [1.0, 2.0, 3.0, ..., 1024.0]
            let testInputBuf = device.makeBuffer(length: hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared)!
            let testOutputBuf = device.makeBuffer(length: hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared)!
            let testPtr = testInputBuf.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
            for i in 0..<hiddenSize {
                testPtr[i] = Float16(Float(i + 1))  // 1.0, 2.0, ..., 1024.0
            }
            // Use ln_f weights for the test (already loaded)
            let lnW = try weight("ln_f.weight")
            let lnB = try weight("ln_f.bias")

            guard let testCmd = commandQueue.makeCommandBuffer(),
                  let testEnc = testCmd.makeComputeCommandEncoder() else {
                NSLog("[MetalLMBackend] TEST: FAILED to create command buffer")
                throw MetalLMError.commandBufferFailed
            }
            testEnc.setComputePipelineState(layerNormPipeline)
            testEnc.setBuffer(testInputBuf, offset: 0, index: 0)
            testEnc.setBuffer(lnW, offset: 0, index: 1)
            testEnc.setBuffer(lnB, offset: 0, index: 2)
            testEnc.setBuffer(testOutputBuf, offset: 0, index: 3)
            var d = UInt32(hiddenSize)
            var eps: Float16 = Float16(1e-5)
            testEnc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)
            testEnc.setBytes(&eps, length: MemoryLayout<Float16>.size, index: 5)
            let tg = MTLSize(width: 256, height: 1, depth: 1)
            let ntg = MTLSize(width: 1, height: 1, depth: 1)
            testEnc.dispatchThreadgroups(ntg, threadsPerThreadgroup: tg)
            testEnc.endEncoding()
            testCmd.commit()
            testCmd.waitUntilCompleted()

            // Check output
            let outPtr = testOutputBuf.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
            var outAbsmax: Float = 0
            var outSample0: Float = 0
            for i in 0..<hiddenSize {
                let v = abs(Float(outPtr[i]))
                if v > outAbsmax { outAbsmax = v }
                if i < 3 { outSample0 = v }
            }
            NSLog("[MetalLMBackend] TEST: LayerNorm output absmax=%.4f sample0=%.4f (expected absmax ~0.1-2 for normalized output)", outAbsmax, outSample0)

            // Test GEMM_NT in isolation: A=[1,1024] @ B^T=[1024,3072] → C=[1,3072]
            NSLog("[MetalLMBackend] TEST: Running GEMM_NT isolation test...")
            let testAbuf = device.makeBuffer(length: hiddenSize * MemoryLayout<Float16>.size, options: .storageModeShared)!
            let testCbuf = device.makeBuffer(length: gpt2NumHeads * headDim * 3 * MemoryLayout<Float16>.size, options: .storageModeShared)!
            let testAptr = testAbuf.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
            for i in 0..<hiddenSize {
                testAptr[i] = Float16(1.0)  // All ones
            }
            let cAttnW = try weight("h.0.attn.c_attn.weight")
            guard let gemmCmd = commandQueue.makeCommandBuffer(),
                  let gemmEnc = gemmCmd.makeComputeCommandEncoder() else {
                NSLog("[MetalLMBackend] TEST: GEMM FAILED to create command buffer")
                throw MetalLMError.commandBufferFailed
            }
            gemmEnc.setComputePipelineState(gemmNTPipeline)
            gemmEnc.setBuffer(testAbuf, offset: 0, index: 0)
            gemmEnc.setBuffer(cAttnW, offset: 0, index: 1)
            gemmEnc.setBuffer(testCbuf, offset: 0, index: 2)
            var m: UInt32 = 1, n: UInt32 = 3072, k: UInt32 = 1024
            gemmEnc.setBytes(&m, length: MemoryLayout<UInt32>.size, index: 3)
            gemmEnc.setBytes(&n, length: MemoryLayout<UInt32>.size, index: 4)
            gemmEnc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 5)
            let ntgX = (3072 + 15) / 16
            let ntgY = (1 + 15) / 16
            let tgSz = MTLSize(width: 16, height: 16, depth: 1)
            let ntgSz = MTLSize(width: ntgX, height: ntgY, depth: 1)
            gemmEnc.dispatchThreadgroups(ntgSz, threadsPerThreadgroup: tgSz)
            gemmEnc.endEncoding()
            gemmCmd.commit()
            gemmCmd.waitUntilCompleted()

            let cPtr = testCbuf.contents().bindMemory(to: Float16.self, capacity: 3072)
            var cAbsmax: Float = 0
            for i in 0..<3072 {
                let v = abs(Float(cPtr[i]))
                if v > cAbsmax { cAbsmax = v }
            }
            NSLog("[MetalLMBackend] TEST: GEMM_NT output absmax=%.4f (expected ~0.1-10)", cAbsmax)

        } catch {
            NSLog("[MetalLMBackend] TEST: FAILED with error: \(error)")
        }
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

    /// Prefill forward pass: processes all seqLen positions through self-attention + FFN.
    ///
    /// For each layer l = 0..numLayers-1:
    ///   For each position p = 0..seqLen-1 (sequential):
    ///     1. Read inputsEmbds[p] → hidden (single position slice)
    ///     2. Pre-LN → ln1_out
    ///     3. QKV proj (single position)
    ///     4. Unpack → prefillQ/K/V[p]
    ///     5. RoPE on prefillQ[p], prefillK[p]
    ///     6. Write K/V[p] → KV cache
    ///     7. GroupQueryAttn (all seqLen positions at once) → attn_out[p]
    ///     8. O_proj + residual_add + post-LN + FFN + residual_add
    ///   Hidden for position p+1 at layer l = hidden from position p at layer l
    ///
    /// After all layers: final LN + lm_head → logits.
    ///
    /// The key difference from forward() is step 7: group_query_attention uses ALL
    /// seqLen positions for self-attention, correctly populating the KV cache for
    /// subsequent decode steps.
    public func forwardPrefill(
        inputsEmbds: MTLBuffer,
        seqLen: Int,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        NSLog("[MetalLMBackend] forwardPrefill: seqLen=%d, starting", seqLen)

        // Reusable buffer for hidden states across all positions: [1, seqLen, 1024] half
        // We allocate once at prefill start and reuse throughout.
        // Note: This adds one-time allocation cost but avoids per-layer reallocation.
        let hiddenSizeBytes = seqLen * hiddenSize * MemoryLayout<Float16>.size
        let allHiddenBuf = UnsafeMutablePointer<Float16>.allocate(capacity: seqLen * hiddenSize)
        defer { allHiddenBuf.deallocate() }

        // Copy initial hidden states from inputsEmbds
        memcpy(allHiddenBuf,
               inputsEmbds.contents().bindMemory(to: Float16.self, capacity: seqLen * hiddenSize),
               hiddenSizeBytes)

        // Use a temporary MTLBuffer wrapping our hidden states
        let allHiddenMTLBuf = device.makeBuffer(
            length: hiddenSizeBytes,
            options: .storageModeShared
        )!

        // Per-layer processing: all positions go through one layer together
        for layer in 0..<numLayers {
            NSLog("[MetalLMBackend] forwardPrefill: layer %d/%d, computing QKV for all positions", layer, numLayers)

            // ── PHASE 1: Pre-LN + QKV for all positions ─────────────────────
            // For each position, compute pre-LN and QKV projection, store in prefill buffers
            for pos in 0..<seqLen {
                let byteOffset = pos * hiddenSize * MemoryLayout<Float16>.size

                // Copy hidden[pos] to ln1OutBuffer (single-position work buffer)
                memcpy(ln1OutBuffer.contents(),
                       allHiddenBuf.advanced(by: pos * hiddenSize),
                       hiddenSize * MemoryLayout<Float16>.size)

                // Pre-LayerNorm
                try runLayerNorm(
                    input: ln1OutBuffer,
                    gamma: weight("h.\(layer).ln_1.weight"),
                    beta: weight("h.\(layer).ln_1.bias"),
                    output: ln1OutBuffer,
                    dim: hiddenSize,
                    commandBuffer: commandBuffer
                )

                // QKV projection
                try runGEMM_NT(
                    A: ln1OutBuffer,
                    B: weight("h.\(layer).attn.c_attn.weight"),
                    C: qkvBuffer,
                    M: 1,
                    N: gpt2NumHeads * headDim * 3,
                    K: hiddenSize,
                    commandBuffer: commandBuffer
                )

                // Unpack qkvBuffer → single-position qBuffer/kWriteBuffer/vWriteBuffer
                unpackQKV()

                // Apply RoPE to Q and K at this position (in-place on single-position buffers)
                try applyRoPELayer(
                    layer: layer,
                    q: qBuffer,
                    k: kWriteBuffer,
                    seqPos: pos,
                    commandBuffer: commandBuffer
                )

                // Copy rotated Q/K/V to prefill buffers at position 'pos'
                try unpackQKVToPrefill(position: pos, seqLen: seqLen)
            }

            // ── PHASE 2: Write K/V for all positions to KV cache ─────────────
            for pos in 0..<seqLen {
                writeKVPairs(layer: layer, kvWriteOffset: pos)
            }

            // ── PHASE 4: Group Query Attention (all positions at once) ─────────
            NSLog("[MetalLMBackend] forwardPrefill: layer %d, group query attention", layer)
            try runPrefillAttention(
                seqLen: seqLen,
                commandBuffer: commandBuffer
            )

            // ── PHASE 5: For each position: O_proj + residual + postLN + FFN ──
            for pos in 0..<seqLen {
                let byteOffset = pos * hiddenSize * MemoryLayout<Float16>.size

                // Extract attn_out[pos] from prefillAttnOutBuffer → attnOutBuffer
                // prefillAttnOutBuffer is [1, numHeads, seqLen, headDim] float32
                // attn_out at 'pos' starts at offset pos * numHeads * headDim * 4
                // attnOutBuffer is [1, numHeads, 1, headDim] float16
                // Need float32 → float16 conversion
                let attnSrcOffset = pos * gpt2NumHeads * headDim * MemoryLayout<Float32>.size
                let attnSrcPtr = prefillAttnOutBuffer.contents()
                    .advanced(by: attnSrcOffset)
                    .bindMemory(to: Float32.self, capacity: gpt2NumHeads * headDim)
                let attnDstPtr = attnOutBuffer.contents().bindMemory(to: Float16.self, capacity: gpt2NumHeads * headDim)
                for i in 0..<(gpt2NumHeads * headDim) {
                    attnDstPtr[i] = Float16(attnSrcPtr[i])
                }

                // O projection → residualBuffer
                try runGEMM_NT(
                    A: attnOutBuffer,
                    B: weight("h.\(layer).attn.c_proj.weight"),
                    C: residualBuffer,
                    M: 1,
                    N: hiddenSize,
                    K: gpt2NumHeads * headDim,
                    commandBuffer: commandBuffer
                )

                // Residual add: residual = hidden + attn_out
                memcpy(ln1OutBuffer.contents(),
                       allHiddenBuf.advanced(by: pos * hiddenSize),
                       hiddenSize * MemoryLayout<Float16>.size)
                try runResidualAdd(
                    a: ln1OutBuffer,
                    b: residualBuffer,
                    output: residualBuffer,
                    size: hiddenSize,
                    commandBuffer: commandBuffer
                )

                // Post-LayerNorm
                try runLayerNorm(
                    input: residualBuffer,
                    gamma: weight("h.\(layer).ln_2.weight"),
                    beta: weight("h.\(layer).ln_2.bias"),
                    output: ln2OutBuffer,
                    dim: hiddenSize,
                    commandBuffer: commandBuffer
                )

                // FFN gate + GELU
                try runGEMM_NT(
                    A: ln2OutBuffer,
                    B: weight("h.\(layer).mlp.c_fc.weight"),
                    C: ffnInterimBuffer,
                    M: 1,
                    N: intermediateSize,
                    K: hiddenSize,
                    commandBuffer: commandBuffer
                )
                try runTanhGelu(
                    input: ffnInterimBuffer,
                    output: ffnInterimBuffer,
                    size: intermediateSize,
                    commandBuffer: commandBuffer
                )

                // FFN up projection
                try runGEMM_NT(
                    A: ffnInterimBuffer,
                    B: weight("h.\(layer).mlp.c_proj.weight"),
                    C: ffnOutBuffer,
                    M: 1,
                    N: hiddenSize,
                    K: intermediateSize,
                    commandBuffer: commandBuffer
                )

                // Residual add: hidden = residual + ffn_hidden
                try runResidualAdd(
                    a: residualBuffer,
                    b: ffnOutBuffer,
                    output: residualBuffer,
                    size: hiddenSize,
                    commandBuffer: commandBuffer
                )

                // Write updated hidden state for position 'pos'
                memcpy(allHiddenBuf.advanced(by: pos * hiddenSize),
                       residualBuffer.contents(),
                       hiddenSize * MemoryLayout<Float16>.size)
            }
        }

        NSLog("[MetalLMBackend] forwardPrefill: final LN + lm_head")
        // Final LayerNorm + LM head on the last position's hidden state
        let lastPosOffset = (seqLen - 1) * hiddenSize * MemoryLayout<Float16>.size
        memcpy(ln1OutBuffer.contents(),
               allHiddenBuf.advanced(by: (seqLen - 1) * hiddenSize),
               hiddenSize * MemoryLayout<Float16>.size)

        return try finalForward(hidden: ln1OutBuffer, commandBuffer: commandBuffer)
    }

    public func reset() async {
        await kvCache.reset()
    }

    /// Return the absmax of the last layer's hidden state (residualBuffer).
    /// Uses a fresh command buffer — safe to call after any committed+waited pass.
    public func getLastHiddenStateAbsmax() -> Float {
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else { return 0 }
        // No-op kernel just to force GPU sync and make buffer contents visible
        enc.setComputePipelineState(residualAddPipeline)
        enc.setBuffer(residualBuffer, offset: 0, index: 0)
        enc.setBuffer(residualBuffer, offset: 0, index: 1)
        enc.setBuffer(residualBuffer, offset: 0, index: 2)
        var s = UInt32(hiddenSize)
        enc.setBytes(&s, length: MemoryLayout<UInt32>.size, index: 3)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()

        let hPtr = residualBuffer.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
        var absmax: Float = 0
        for i in 0..<hiddenSize {
            let v = abs(Float(hPtr[i]))
            if v > absmax { absmax = v }
        }
        return absmax
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
        try runGEMM_NT(
            A: ln1OutBuffer,
            B: weight("h.\(layer).attn.c_attn.weight"),
            C: qkvBuffer,
            M: 1,
            N: gpt2NumHeads * headDim * 3,
            K: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 3. Unpack Q/K/V from qkvBuffer [1, 1, 3072]
        unpackQKV()

        // 4. Apply RoPE to Q and K
        try applyRoPELayer(
            layer: layer,
            q: qBuffer,
            k: kWriteBuffer,
            seqPos: kvWriteOffset,
            commandBuffer: commandBuffer
        )

        // 5. Write K/V to KV cache
        writeKVPairs(layer: layer, kvWriteOffset: kvWriteOffset)

        // 6. Attention
        try runAttention(
            q: qBuffer,
            layer: layer,
            kvReadLength: kvReadLength,
            commandBuffer: commandBuffer
        )

        // 7. O projection
        try runGEMM_NT(
            A: attnOutBuffer,
            B: weight("h.\(layer).attn.c_proj.weight"),
            C: residualBuffer,
            M: 1,
            N: hiddenSize,
            K: gpt2NumHeads * headDim,
            commandBuffer: commandBuffer
        )

        // 8. Residual add: residual = hidden + o
        try runResidualAdd(
            a: hidden,
            b: residualBuffer,
            output: residualBuffer,
            size: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 9. Post-LayerNorm
        try runLayerNorm(
            input: residualBuffer,
            gamma: weight("h.\(layer).ln_2.weight"),
            beta: weight("h.\(layer).ln_2.bias"),
            output: ln2OutBuffer,
            dim: hiddenSize,
            commandBuffer: commandBuffer
        )

        // 10. FFN gate + GELU
        try runGEMM_NT(
            A: ln2OutBuffer,
            B: weight("h.\(layer).mlp.c_fc.weight"),
            C: ffnInterimBuffer,
            M: 1,
            N: intermediateSize,
            K: hiddenSize,
            commandBuffer: commandBuffer
        )
        try runTanhGelu(
            input: ffnInterimBuffer,
            output: ffnInterimBuffer,
            size: intermediateSize,
            commandBuffer: commandBuffer
        )

        // 11. FFN up projection
        try runGEMM_NT(
            A: ffnInterimBuffer,
            B: weight("h.\(layer).mlp.c_proj.weight"),
            C: ffnOutBuffer,
            M: 1,
            N: hiddenSize,
            K: intermediateSize,
            commandBuffer: commandBuffer
        )

        // 12. Residual add: hidden = residual + ffn_hidden
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
    /// NOTE: Caller must commit+wait the commandBuffer AFTER this returns
    ///       before reading the logits buffer.
    public func finalForward(
        hidden: MTLBuffer,
        commandBuffer: MTLCommandBuffer
    ) throws -> MTLBuffer {
        // Read lm_head weight stats directly (CPU-side, no GPU sync needed)
        let lmBuf = try weight("lm_head.weight")
        let lmPtr = lmBuf.contents().bindMemory(to: Float16.self, capacity: vocabSize * hiddenSize)
        var lmMaxF: Float = 0
        for i in 0..<(vocabSize * hiddenSize) {
            let v = abs(Float(lmPtr[i]))
            if v > lmMaxF { lmMaxF = v }
        }

        // Final LayerNorm (encoded into caller's commandBuffer)
        try runLayerNorm(
            input: hidden,
            gamma: weight("ln_f.weight"),
            beta: weight("ln_f.bias"),
            output: finalLnOutBuffer,
            dim: hiddenSize,
            commandBuffer: commandBuffer
        )

        // LM head: logits = gemm_nt(ln_f_out, lm_head.weight)
        // Encoded on the SAME caller's commandBuffer (no sync here — caller manages commit+wait)
        try runGEMM_NT(
            A: finalLnOutBuffer,
            B: lmBuf,
            C: logitsBuffer,
            M: 1,
            N: vocabSize,
            K: hiddenSize,
            commandBuffer: commandBuffer
        )

        // NOTE: GPU sync (commit+wait) is caller's responsibility after forward() returns.
        // We do NOT sync here so the caller can chain multiple forwards on the same buffer.
        // The hidden, finalLnOut, and logitsBuffer will have valid contents after commit+wait.

        return logitsBuffer
    }

    /// Inspect final hidden state and logits after a committed+waited forward pass.
    /// MUST be called after the command buffer has been committed and waited.
    /// This reads GPU buffer contents on the CPU — safe to call multiple times.
    public func inspectFinalState(
        hidden: MTLBuffer,
        logitsBuf: MTLBuffer
    ) {
        // hidden: last layer output (residualBuffer)
        let hiddenPtr = hidden.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
        var hiddenMaxF: Float = 0
        for i in 0..<hiddenSize {
            let v = abs(Float(hiddenPtr[i]))
            if v > hiddenMaxF { hiddenMaxF = v }
        }

        // ln_f output (finalLnOutBuffer — output of final LayerNorm)
        let lnOutPtr = finalLnOutBuffer.contents().bindMemory(to: Float16.self, capacity: hiddenSize)
        var lnOutMaxF: Float = 0
        for i in 0..<hiddenSize {
            let v = abs(Float(lnOutPtr[i]))
            if v > lnOutMaxF { lnOutMaxF = v }
        }

        // logits
        let logitsPtr = logitsBuf.contents().bindMemory(to: Float.self, capacity: vocabSize)
        var logitsMaxF: Float = 0
        for i in 0..<vocabSize {
            let v = abs(logitsPtr[i])
            if v > logitsMaxF { logitsMaxF = v }
        }
        NSLog("[MetalLMBackend] inspect: hidden absmax=%.6f, ln_f absmax=%.6f, logits absmax=%.6f", hiddenMaxF, lnOutMaxF, logitsMaxF)
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

        // layer_norm: 256 threads × 4 elements/thread = 1024 (dim)
        // Parallel reduction in threadgroup memory for mean/variance
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
    /// attention_decode_step kernel: Grid: (16, 64), Threadgroup: (16, 4)
    /// GPT2NoEmbed standard MHA: 16 Q heads, 16 KV heads, ratio=1.
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
        enc.setBuffer(q, offset: 0, index: 0)          // Q [16, 64]
        enc.setBuffer(kBuf, offset: 0, index: 1)      // K [16, maxSeq, 64]
        enc.setBuffer(vBuf, offset: 0, index: 2)     // V [16, maxSeq, 64]
        enc.setBuffer(attnOutBuffer, offset: 0, index: 3)  // output [16, 64]

        var kvLen = UInt32(kvReadLength)
        var maxSeq = UInt32(maxSeqLen)
        enc.setBytes(&kvLen, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&maxSeq, length: MemoryLayout<UInt32>.size, index: 5)

        // Grid: MTLSize(width: 16, height: 64, depth: 1)
        // Threadgroup: MTLSize(width: 16, height: 4, depth: 1)
        let threadsPerGroup = MTLSize(width: 16, height: 4, depth: 1)
        let numThreadGroups = MTLSize(width: 16, height: 64, depth: 1)
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
    /// Q: [16, 64] half (modified in-place after kernel writes back)
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

        // Grid: (num_heads_total, head_dim/2) = (32, 32)
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

    /// Unpack fused qkvBuffer [1, 1, 3072] into prefill Q/K/V buffers at a specific position.
    /// GPT2NoEmbed: qkv = [1, 1, 3072] = 16*64*3 = Q + K + V
    /// Pre-fill buffers: [1, numHeads, seqLen, headDim] half
    /// At position 'pos': copy Q/K/V to prefillQ/K/V[h, pos, d] = qkv[h*64 + d], h*64+d, h*64+d
    private func unpackQKVToPrefill(position: Int, seqLen: Int) throws {
        let qkvPtr = qkvBuffer.contents().bindMemory(to: Float16.self, capacity: 3072)
        let preQPtr = prefillQBuffer.contents().bindMemory(to: Float16.self, capacity: 1 * gpt2NumHeads * maxSeqLen * headDim)
        let preKPtr = prefillKBuffer.contents().bindMemory(to: Float16.self, capacity: 1 * gpt2NumHeads * maxSeqLen * headDim)
        let preVPtr = prefillVBuffer.contents().bindMemory(to: Float16.self, capacity: 1 * gpt2NumHeads * maxSeqLen * headDim)

        for h in 0..<gpt2NumHeads {
            for d in 0..<headDim {
                let qkv_idx = h * headDim + d
                // prefill buffer at [h, pos, d]
                let pre_idx = h * seqLen * headDim + position * headDim + d
                preQPtr[pre_idx] = qkvPtr[qkv_idx]                          // Q
                preKPtr[pre_idx] = qkvPtr[1024 + qkv_idx]                   // K
                preVPtr[pre_idx] = qkvPtr[2048 + qkv_idx]                   // V
            }
        }
    }

    /// Apply RoPE to prefill Q and K buffers for all seqLen positions.
    /// The prefill buffers are [1, numHeads, seqLen, headDim].
    /// We apply RoPE per-position using the same ropeKernelPipeline but need
    /// to handle the different buffer layout.
    private func applyRoPEPrefill(
        layer: Int,
        seqLen: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(ropeKernelPipeline)
        enc.setBuffer(prefillQBuffer, offset: 0, index: 0)
        enc.setBuffer(prefillKBuffer, offset: 0, index: 1)
        enc.setBuffer(ropeCosBuffer, offset: 0, index: 2)
        enc.setBuffer(ropeSinBuffer, offset: 0, index: 3)

        var numHeads = UInt32(gpt2NumHeads)
        var seq = UInt32(seqLen)
        var hd = UInt32(headDim)
        enc.setBytes(&numHeads, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&seq, length: MemoryLayout<UInt32>.size, index: 5)
        enc.setBytes(&hd, length: MemoryLayout<UInt32>.size, index: 6)

        // For prefill, we apply RoPE one position at a time using the standard kernel
        // by only considering the relevant slice. Grid: (numHeads, seqLen/2) = (16, 82)
        let threadsPerGroup = MTLSize(width: 16, height: 4, depth: 1)
        let numThreadGroups = MTLSize(
            width: (gpt2NumHeads + 15) / 16,
            height: ((seqLen / 2) + 3) / 4,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
    }

    /// Run group_query_attention for the full sequence (prefill self-attention).
    /// Uses prefillQ/K/V buffers containing all positions' Q/K/V after RoPE.
    /// Writes output to prefillAttnOutBuffer [1, numHeads, seqLen, headDim] float32.
    /// Then reads back and writes per-position results to attnOutBuffer.
    private func runPrefillAttention(
        seqLen: Int,
        commandBuffer: MTLCommandBuffer
    ) throws {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        defer { enc.endEncoding() }

        enc.setComputePipelineState(prefillAttentionPipeline)
        enc.setBuffer(prefillQBuffer, offset: 0, index: 0)
        enc.setBuffer(prefillKBuffer, offset: 0, index: 1)
        enc.setBuffer(prefillVBuffer, offset: 0, index: 2)
        enc.setBuffer(prefillAttnOutBuffer, offset: 0, index: 3)

        var seqLenU = UInt32(seqLen)
        enc.setBytes(&seqLenU, length: MemoryLayout<UInt32>.size, index: 4)

        // Grid: (numHeads=16, seqLen) = (16, 164) → one thread per (q_head, seq_pos)
        // Each thread computes all head_dim values for that (q_head, seq_pos)
        let threadsPerGroup = MTLSize(width: 16, height: 4, depth: 1)
        let numThreadGroups = MTLSize(
            width: gpt2NumHeads,
            height: seqLen,
            depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
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
        prefillAttentionPipeline = try makePipeline("group_query_attention")
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

        // Prefill attention: [1, numHeads, maxSeqLen, headDim] half/float
        let prefillQSize = 1 * gpt2NumHeads * maxSeqLen * headDim * fp16
        let prefillAttnOutSize = 1 * gpt2NumHeads * maxSeqLen * headDim * fp32
        prefillQBuffer = device.makeBuffer(length: prefillQSize, options: .storageModeShared)!
        prefillKBuffer = device.makeBuffer(length: prefillQSize, options: .storageModeShared)!
        prefillVBuffer = device.makeBuffer(length: prefillQSize, options: .storageModeShared)!
        prefillAttnOutBuffer = device.makeBuffer(length: prefillAttnOutSize, options: .storageModeShared)!
    }

    private func precomputeRoPE() throws {
        NSLog("[MetalLMBackend] precomputeRoPE: START")
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        NSLog("[MetalLMBackend] precomputeRoPE: commandBuffer+encoder created OK, encoding...")

        guard let ropeLutFunc = library.makeFunction(name: "compute_rope_lut") else {
            throw MetalLMError.kernelNotFound("compute_rope_lut")
        }
        NSLog("[MetalLMBackend] precomputeRoPE: JIT compiling compute_rope_lut pipeline...")
        let ropeLutPipeline = try device.makeComputePipelineState(function: ropeLutFunc)
        NSLog("[MetalLMBackend] precomputeRoPE: pipeline compiled OK, encoding kernel...")

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
        NSLog("[MetalLMBackend] precomputeRoPE: dispatching...")

        // MUST end encoding BEFORE commit — Metal asserts if you commit while encoding is in progress.
        // defer is removed and endEncoding is called explicitly before commit.
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        NSLog("[MetalLMBackend] precomputeRoPE: DONE")
    }

    private func precomputeCausalMask() throws {
        NSLog("[MetalLMBackend] precomputeCausalMask: START")
        guard let cmd = commandQueue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            throw MetalLMError.commandBufferFailed
        }
        NSLog("[MetalLMBackend] precomputeCausalMask: commandBuffer+encoder created OK, encoding...")

        guard let maskFunc = library.makeFunction(name: "causal_mask_kernel") else {
            throw MetalLMError.kernelNotFound("causal_mask_kernel")
        }
        let maskPipeline = try device.makeComputePipelineState(function: maskFunc)

        enc.setComputePipelineState(maskPipeline)
        enc.setBuffer(causalMaskBuffer, offset: 0, index: 0)

        var seqLen = UInt32(maxSeqLen)
        enc.setBytes(&seqLen, length: MemoryLayout<UInt32>.size, index: 1)

        let totalThreads = maxSeqLen * maxSeqLen
        // iOS GPU max threadsPerThreadgroup = 512; use 512 for the 2D mask
        let threadsPerGroup = MTLSize(width: 512, height: 1, depth: 1)
        let numThreadGroups = MTLSize(
            width: (totalThreads + 511) / 512,
            height: 1, depth: 1
        )
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        NSLog("[MetalLMBackend] precomputeCausalMask: dispatching...")

        // MUST end encoding BEFORE commit — Metal asserts if you commit while encoding is in progress.
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        NSLog("[MetalLMBackend] precomputeCausalMask: DONE")
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

        // DIAGNOSTIC: Verify key weights are non-zero
        let keyWeights = ["ln_f.weight", "ln_f.bias", "lm_head.weight",
                          "h.0.ln_1.weight", "h.0.ln_1.bias",
                          "h.0.attn.c_attn.weight", "h.0.attn.c_proj.weight",
                          "h.0.mlp.c_fc.weight", "h.0.mlp.c_proj.weight"]
        for name in keyWeights {
            if let buf = weightBuffers[name] {
                let ptr = buf.contents().bindMemory(to: Float16.self, capacity: buf.length / MemoryLayout<Float16>.size)
                let count = buf.length / MemoryLayout<Float16>.size
                var absmax: Float = 0
                var sample0: Float = 0
                for i in 0..<count {
                    let v = abs(Float(ptr[i]))
                    if v > absmax { absmax = v }
                    if i < 3 { sample0 = v }
                }
                NSLog("[MetalLMBackend] loadWeights: %@ absmax=%.4f sample0=%.4f count=%d", name, absmax, sample0, count)
            } else {
                NSLog("[MetalLMBackend] loadWeights: %@ NOT FOUND", name)
            }
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

