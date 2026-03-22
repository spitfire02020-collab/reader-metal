import Foundation
import Metal
import MetalPerformanceShadersGraph

// MARK: - GEMM Utilities via MPSGraph

/// MPSGraph-based GEMM operations.
/// All matrices are [S, M] or [S, N] row-major Float16.
/// iOS 18+ required (MPSGraph).
final class MPSGEMM {
    let device: MTLDevice
    let graph: MPSGraph

    init(device: MTLDevice) {
        self.device = device
        self.graph = MPSGraph()
    }

    /// C = A @ B^T (transpose B)
    /// A: [S, M] row-major, B: [N, M] row-major → C: [S, N] row-major
    func matmulTransposeB(
        commandBuffer: MTLCommandBuffer,
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        batch: Int, S: Int, M: Int, N: Int
    ) {
        let shapeA = [NSNumber(value: S), NSNumber(value: M)]
        let shapeB = [NSNumber(value: N), NSNumber(value: M)]
        let shapeC = [NSNumber(value: S), NSNumber(value: N)]

        let tensorA = graph.variable(
            with: MPSGraphTensorData(A),
            shape: shapeA,
            name: "A"
        )
        let tensorB = graph.variable(
            with: MPSGraphTensorData(B),
            shape: shapeB,
            name: "B"
        )

        // B^T: transpose B from [N, M] to [M, N]
        let tensorB_T = graph.transpose(tensorB, axes: [1, 0], name: "B_T")

        // C = A @ B^T
        let tensorC = graph.matmul(
            tensorA,
            tensorB_T,
            name: "C"
        )

        let outputTensor = graph.variable(
            with: MPSGraphTensorData(C),
            shape: shapeC,
            name: "C_out"
        )

        // Bind output
        graph.bind(outputTensor, to: C)

        let executable = graph.compile(
            with: device,
            primaryBatchSize: batch,
            minimumPower: .any,
            cacheDirectory: nil
        )

        executable.run(
            with: commandBuffer,
            inputs: [MPSGraphTensorData(A)],
            intermediateTensors: nil,
            outputs: [MPSGraphTensorData(C)],
            options: [],
            queue: nil
        )
    }

    /// C = A @ B (no transpose)
    /// A: [S, M] row-major, B: [M, N] row-major → C: [S, N] row-major
    func matmul(
        commandBuffer: MTLCommandBuffer,
        A: MTLBuffer, B: MTLBuffer, C: MTLBuffer,
        batch: Int, S: Int, M: Int, N: Int
    ) {
        let shapeA = [NSNumber(value: S), NSNumber(value: M)]
        let shapeB = [NSNumber(value: M), NSNumber(value: N)]
        let shapeC = [NSNumber(value: S), NSNumber(value: N)]

        let tensorA = graph.variable(
            with: MPSGraphTensorData(A),
            shape: shapeA,
            name: "A"
        )
        let tensorB = graph.variable(
            with: MPSGraphTensorData(B),
            shape: shapeB,
            name: "B"
        )

        let tensorC = graph.matmul(tensorA, tensorB, name: "C")

        graph.bind(tensorC, to: C)

        let executable = graph.compile(
            with: device,
            primaryBatchSize: batch,
            minimumPower: .any,
            cacheDirectory: nil
        )

        executable.run(
            with: commandBuffer,
            inputs: [MPSGraphTensorData(A)],
            intermediateTensors: nil,
            outputs: [MPSGraphTensorData(C)],
            options: [],
            queue: nil
        )
    }
}

// MARK: - SDPA (Multi-head Attention with Causal Mask)

/// Scaled Dot-Product Attention with causal masking.
/// Q: [B, S, H*D], K: [B, S_full, H*D], V: [B, S_full, H*D]
/// Returns: [B, S, H*D]
/// iOS 18+ required (MPSGraph).
final class MPSDPA {
    let device: MTLDevice
    let graph: MPSGraph

    init(device: MTLDevice) {
        self.device = device
        self.graph = MPSGraph()
    }

    func forward(
        commandBuffer: MTLCommandBuffer,
        Q: MTLBuffer, K: MTLBuffer, V: MTLBuffer,
        B: Int, S: Int, S_full: Int, H: Int, D: Int,
        output: MTLBuffer
    ) {
        // Reshape [B, S, H*D] → [B, H, S, D]
        let qShapeBHSD = [NSNumber(value: B), NSNumber(value: H), NSNumber(value: S), NSNumber(value: D)]
        let kShapeBHSD = [NSNumber(value: B), NSNumber(value: H), NSNumber(value: S_full), NSNumber(value: D)]

        let qFlat = graph.variable(
            with: MPSGraphTensorData(Q),
            shape: [NSNumber(value: B), NSNumber(value: S), NSNumber(value: H * D)],
            name: "Q_flat"
        )
        let kFlat = graph.variable(
            with: MPSGraphTensorData(K),
            shape: [NSNumber(value: B), NSNumber(value: S_full), NSNumber(value: H * D)],
            name: "K_flat"
        )
        let vFlat = graph.variable(
            with: MPSGraphTensorData(V),
            shape: [NSNumber(value: B), NSNumber(value: S_full), NSNumber(value: H * D)],
            name: "V_flat"
        )

        // Reshape to [B, H, S, D]
        let qTensor = graph.reshape(qFlat, shape: qShapeBHSD, name: "Q")
        let kTensor = graph.reshape(kFlat, shape: kShapeBHSD, name: "K")
        let vTensor = graph.reshape(vFlat, shape: kShapeBHSD, name: "V")

        // ----- Causal mask -----
        // mask[i, j] = 1 if j <= i (can attend), else 0
        // For decode step (S=1): position 0 attends to all S_full positions
        var maskValues = [Float](repeating: 0, count: S * S_full)
        if S == 1 {
            // Decode step: single query, attend to all key positions
            for j in 0..<S_full {
                maskValues[j] = 1.0
            }
        } else {
            // Prefix: causal mask
            for i in 0..<S {
                for j in 0..<S_full {
                    maskValues[i * S_full + j] = (j <= i) ? 1.0 : 0.0
                }
            }
        }

        let maskConst = graph.constant(maskValues, shape: [NSNumber(value: S), NSNumber(value: S_full)], name: "causalMask")

        // Expand mask to [1, 1, S, S_full] for broadcasting over B and H
        let maskExpanded = graph.expandDims(maskConst, axes: [0, 1], name: "maskExpanded")

        // ----- Scaled dot-product attention -----
        // scale = 1 / sqrt(D)
        let scaleConst = graph.constant(Float(1.0 / sqrt(Double(D))), name: "scale")

        // Q @ K^T: [B,H,S,D] @ [B,H,D,S_full] → [B,H,S,S_full]
        let qkt = graph.matmul(qTensor, kTensor, name: nil, resulttranspose: false, lefttranspose: false, righttranspose: true)

        let scaledQKT = graph.multiply(qkt, scaleConst, name: "scaledQKT")

        // Apply mask: masked positions → -inf
        let negInf = graph.constant(-Float.infinity, name: "negInf")
        // select(condition, trueTensor, falseTensor)
        // Where maskExpanded == 1 → keep scaledQKT, else → -inf
        let maskedQKT = graph.select(
            condition: maskExpanded,
            trueTensor: scaledQKT,
            falseTensor: negInf,
            name: "maskedQKT"
        )

        // Softmax over last axis (S_full)
        let attnWeights = graph.softMax(maskedQKT, axis: NSNumber(value: S_full - 1), name: "attnWeights")

        // attnWeights @ V: [B,H,S,S_full] @ [B,H,S_full,D] → [B,H,S,D]
        let attnResult = graph.matmul(attnWeights, vTensor, name: nil)

        // Reshape [B,H,S,D] → [B,S,H*D]
        let outputShapeFlat = [NSNumber(value: B), NSNumber(value: S), NSNumber(value: H * D)]
        let outputFlat = graph.reshape(attnResult, shape: outputShapeFlat, name: "attnOutputFlat")

        // Bind output to buffer
        graph.bind(outputFlat, to: output)

        // Compile and run
        let executable = graph.compile(
            with: device,
            primaryBatchSize: B,
            minimumPower: .any,
            cacheDirectory: nil
        )

        executable.run(
            with: commandBuffer,
            inputs: [MPSGraphTensorData(Q), MPSGraphTensorData(K), MPSGraphTensorData(V)],
            intermediateTensors: nil,
            outputs: [MPSGraphTensorData(output)],
            options: [],
            queue: nil
        )
    }
}

// MARK: - LayerNorm (Metal compute pipeline)

/// LayerNorm via custom Metal compute kernel (not MPSGraph).
/// y = (x - mean) / sqrt(var + eps) * gamma + beta
final class LayerNormPipeline {
    let device: MTLDevice
    let pipeline: MTLComputePipelineState

    init(device: MTLDevice, library: MTLLibrary) throws {
        self.device = device
        guard let kern = library.makeFunction(name: "layer_norm") else {
            throw MetalLMError.kernelNotFound("layer_norm")
        }
        self.pipeline = try device.makeComputePipelineState(function: kern)
    }

    /// Normalize a [batch, dim] input matrix.
    /// Each row is normalized independently.
    func normalize(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, gamma: MTLBuffer, beta: MTLBuffer, output: MTLBuffer,
        batch: Int, dim: Int
    ) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(pipeline)
        enc.setBuffer(input, offset: 0, index: 0)
        enc.setBuffer(gamma, offset: 0, index: 1)
        enc.setBuffer(beta, offset: 0, index: 2)
        enc.setBuffer(output, offset: 0, index: 3)

        var d = UInt32(dim)
        var eps: Float16 = Float16(1e-5)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&eps, length: MemoryLayout<Float16>.size, index: 5)

        let threadsPerGroup = MTLSize(width: 256, height: 1, depth: 1)
        let numThreadGroups = MTLSize(width: (batch + 255) / 256, height: 1, depth: 1)
        enc.dispatchThreadgroups(numThreadGroups, threadsPerThreadgroup: threadsPerGroup)
        enc.endEncoding()
    }
}
