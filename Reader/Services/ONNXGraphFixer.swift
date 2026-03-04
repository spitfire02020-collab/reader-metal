import Foundation

// MARK: - ONNX Graph Fixer
//
// Fixes three bugs in ONNX models exported by PyTorch / ResembleAI chatterbox-turbo-ONNX:
//
// Bug 1 — GatherBlockQuantized type mismatch (ORT 1.20 iOS)
//   ORT 1.20's GatherBlockQuantized kernel is registered for int4/uint4 ONNX types, but the
//   exported models declare their quantized-weight and zero-point initializers as uint8.
//   Additionally, with uint4 the ONNX packing semantics differ: 2 elements per byte, so
//   the logical dimension count must double.
//
//   Fix:
//     a) Change data_type from 2 (uint8) → 21 (uint4) in the affected TensorProtos.
//     b) Double the last dimension so the element count × 0.5 bytes/element still
//        equals the original byte length (which is already stored in .onnx_data).
//
//   Example — speech_emb_weight_quant:
//     Before: dims=[6563, 512]  dtype=uint8  → 6563×512 = 3,360,256 bytes in .onnx_data
//     After:  dims=[6563,1024]  dtype=uint4  → ⌈6563×1024/2⌉ = 3,360,256 bytes ✓
//
// Bug 2 — Topological sort (PyTorch 2.9 ONNX exporter, older models)
//   ONNX Runtime validates that graph nodes are in topological order.
//   Fix: Kahn's BFS topological sort on the node list.
//
// Bug 3 — GatherBlockQuantized "bits" attribute (ORT 1.20 iOS)
//   ORT 1.20 does not recognise the "bits" attribute on GatherBlockQuantized nodes —
//   it infers the bit-width from the tensor type (uint4 / int4) instead.
//   Models exported targeting newer ORT versions include bits=4 explicitly.
//   Fix: strip the "bits" AttributeProto from every GatherBlockQuantized node.
//
// All fixes operate only on the small .onnx graph file and never touch .onnx_data.
//
// ONNX protobuf fields used:
//   ModelProto.graph          = field 7  (length-delimited)
//   GraphProto.node           = field 1  (repeated, LD)
//   GraphProto.initializer    = field 5  (repeated, LD)
//   NodeProto.input           = field 1  (repeated string)
//   NodeProto.output          = field 2  (repeated string)
//   NodeProto.attribute       = field 5  (repeated AttributeProto, LD)
//   NodeProto.op_type         = field 4  (string)
//   AttributeProto.name       = field 1  (string)
//   TensorProto.dims          = field 1  (repeated int64, non-packed varint or packed LD)
//   TensorProto.data_type     = field 2  (int32, varint)
//   TensorProto.name          = field 8  (string)

enum ONNXGraphFixer {

    // MARK: - Public API

    /// Fix all known issues in an ONNX graph file.
    /// Safe to call repeatedly — writes only when a change was made.
    static func fixIfNeeded(at url: URL) throws {
        let original = try Data(contentsOf: url)
        let fixed = fixModel(original)
        if fixed != original {
            try fixed.write(to: url, options: .atomic)
        }
    }

    // MARK: - Protobuf Varint Helpers

    private static func readVarint(_ data: Data, _ offset: inout Int) -> UInt64 {
        var result: UInt64 = 0
        var shift = 0
        while offset < data.count {
            let byte = UInt64(data[offset])
            offset += 1
            result |= (byte & 0x7F) << shift
            if byte & 0x80 == 0 { break }
            shift += 7
        }
        return result
    }

    private static func encodeVarint(_ value: UInt64) -> Data {
        var v = value
        var out = Data()
        repeat {
            var byte = UInt8(v & 0x7F)
            v >>= 7
            if v != 0 { byte |= 0x80 }
            out.append(byte)
        } while v != 0
        return out
    }

    /// Advance past a field value. Returns the raw bytes skipped (length prefix + content for LD).
    private static func consumeValue(_ data: Data, wireType: Int, _ offset: inout Int) -> Data {
        let start = offset
        switch wireType {
        case 0: _ = readVarint(data, &offset)
        case 1: offset = min(offset + 8, data.count)
        case 2:
            let len = Int(readVarint(data, &offset))
            offset = min(offset + len, data.count)
        case 5: offset = min(offset + 4, data.count)
        default: offset = data.count
        }
        return Data(data[start ..< offset])
    }

    /// Encode a length-delimited field: tag + length + payload.
    private static func encodeLD(field: Int, value: Data) -> Data {
        let tag = encodeVarint(UInt64((field << 3) | 2))
        let len = encodeVarint(UInt64(value.count))
        return tag + len + value
    }

    // MARK: - ModelProto Level

    private static func fixModel(_ data: Data) -> Data {
        var result = Data(capacity: data.count)
        var offset = 0

        while offset < data.count {
            let tagStart = offset
            let tag = readVarint(data, &offset)
            guard offset <= data.count else { break }

            let wireType  = Int(tag & 0x7)
            let fieldNum  = Int(tag >> 3)
            let tagBytes  = Data(data[tagStart ..< offset])

            if fieldNum == 7, wireType == 2 {
                // graph (GraphProto)
                let len       = Int(readVarint(data, &offset))
                let graphData = Data(data[offset ..< min(offset + len, data.count)])
                offset += len
                result += encodeLD(field: 7, value: fixGraph(graphData))
            } else {
                result += tagBytes
                result += consumeValue(data, wireType: wireType, &offset)
            }
        }

        return result
    }

    // MARK: - GraphProto Level

    private static func fixGraph(_ data: Data) -> Data {
        // Phase 1: scan all nodes to find GatherBlockQuantized input names (indices 0 and 3).
        // Those initializers have data_type=uint8 but ORT 1.20 requires uint4 for its kernel.
        let gbqInputNames = findGatherBlockQuantizedInputNames(in: data)

        // Phase 2: collect nodes and initializers for further processing.
        var nodes        = [Data]()       // NodeProto blobs (field 1)
        var initializers = [Data]()       // TensorProto blobs (field 5) — may need patching
        var otherBytes   = Data()         // everything else (field ≠ 1 and ≠ 5)
        var offset       = 0

        while offset < data.count {
            let tagStart = offset
            let tag      = readVarint(data, &offset)
            guard offset <= data.count else { break }

            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)
            let tagBytes = Data(data[tagStart ..< offset])

            if fieldNum == 1, wireType == 2 {
                // node (NodeProto)
                let len      = Int(readVarint(data, &offset))
                let nodeData = Data(data[offset ..< min(offset + len, data.count)])
                offset += len
                nodes.append(nodeData)
            } else if fieldNum == 5, wireType == 2 {
                // initializer (TensorProto)
                let len      = Int(readVarint(data, &offset))
                let initData = Data(data[offset ..< min(offset + len, data.count)])
                offset += len
                initializers.append(initData)
            } else {
                otherBytes += tagBytes
                otherBytes += consumeValue(data, wireType: wireType, &offset)
            }
        }

        guard !nodes.isEmpty else { return data }

        // Phase 3: topologically sort nodes AND strip unsupported "bits" attribute.
        let sorted = topologicalSort(nodes).map { stripGBQBitsAttribute($0) }

        // Phase 4: fix initializer data_types and dims for GatherBlockQuantized inputs.
        let fixedInitializers = initializers.map { initData in
            fixInitializerType(initData, targetNames: gbqInputNames)
        }

        // Phase 5: reassemble graph.
        var result = otherBytes
        for initData in fixedInitializers {
            result += encodeLD(field: 5, value: initData)
        }
        for nodeData in sorted {
            result += encodeLD(field: 1, value: nodeData)
        }
        return result
    }

    // MARK: - GatherBlockQuantized: find input names that need type patching

    /// Scan all nodes; for each GatherBlockQuantized, collect input[0] (quant weight)
    /// and input[3] (zero point) — these need data_type changed from uint8 → uint4
    /// and their last dimension doubled.
    private static func findGatherBlockQuantizedInputNames(in graphData: Data) -> Set<String> {
        var result = Set<String>()
        var offset = 0

        while offset < graphData.count {
            let tag = readVarint(graphData, &offset)
            guard offset <= graphData.count else { break }

            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)

            if fieldNum == 1, wireType == 2 {
                let len      = Int(readVarint(graphData, &offset))
                let nodeData = Data(graphData[offset ..< min(offset + len, graphData.count)])
                offset += len
                let (inputs, opType) = parseNodeInputsAndOpType(nodeData)
                if opType == "GatherBlockQuantized" {
                    if inputs.count > 0, !inputs[0].isEmpty { result.insert(inputs[0]) }
                    if inputs.count > 3, !inputs[3].isEmpty { result.insert(inputs[3]) }
                }
            } else {
                _ = consumeValue(graphData, wireType: wireType, &offset)
            }
        }

        return result
    }

    /// Parse a NodeProto to extract its inputs (field 1) and op_type (field 4).
    private static func parseNodeInputsAndOpType(_ data: Data) -> ([String], String) {
        var inputs = [String]()
        var opType = ""
        var offset = 0

        while offset < data.count {
            let tag      = readVarint(data, &offset)
            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)

            if wireType == 2 {
                let len   = Int(readVarint(data, &offset))
                let bytes = Data(data[offset ..< min(offset + len, data.count)])
                offset   += len
                if let str = String(data: bytes, encoding: .utf8) {
                    switch fieldNum {
                    case 1: inputs.append(str)   // input names
                    case 4: opType = str          // op_type
                    default: break
                    }
                }
            } else {
                _ = consumeValue(data, wireType: wireType, &offset)
            }
        }

        return (inputs, opType)
    }

    // MARK: - TensorProto: fix data_type uint8 → uint4 and double last dim

    /// For a TensorProto blob where name is in targetNames and data_type == 2 (uint8):
    ///   1. Change data_type from 2 (uint8) → 21 (uint4).
    ///   2. Double the last dimension so the element-count×0.5 bytes/element still equals
    ///      the original byte count already stored in .onnx_data.
    ///
    /// Example: [6563, 512] uint8 (3,360,256 bytes) → [6563, 1024] uint4 (3,360,256 bytes).
    private static func fixInitializerType(_ data: Data, targetNames: Set<String>) -> Data {
        guard !targetNames.isEmpty else { return data }

        // Quick scan: get the initializer's name (TensorProto.name = field 8, string).
        let name = parseInitializerName(data)
        guard !name.isEmpty, targetNames.contains(name) else { return data }

        // --- First pass: verify data_type == uint8, count non-packed dims.
        var currentDataType: UInt64 = 0
        var nonPackedDimCount = 0      // how many separate field-1 varint entries exist
        var hasPackedDims     = false  // whether field-1 is encoded as a single LD entry

        var offset = 0
        while offset < data.count {
            let tag = readVarint(data, &offset)
            guard offset <= data.count else { break }
            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)

            if fieldNum == 1, wireType == 0 {
                // non-packed dim (repeated int64 as individual varint)
                _ = readVarint(data, &offset)
                nonPackedDimCount += 1
            } else if fieldNum == 1, wireType == 2 {
                // packed dims (proto3 default for repeated int64)
                let len = Int(readVarint(data, &offset))
                offset = min(offset + len, data.count)
                hasPackedDims = true
            } else if fieldNum == 2, wireType == 0 {
                currentDataType = readVarint(data, &offset)
            } else {
                _ = consumeValue(data, wireType: wireType, &offset)
            }
        }

        guard currentDataType == 2 else { return data } // only fix uint8 → uint4

        // --- Second pass: rewrite with doubled last dim and changed data_type.
        var result = Data(capacity: data.count)
        offset = 0
        var dimsSeenNonPacked = 0

        while offset < data.count {
            let tagStart = offset
            let tag      = readVarint(data, &offset)
            guard offset <= data.count else { break }

            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)
            let tagBytes = Data(data[tagStart ..< offset])

            if fieldNum == 1, wireType == 0 {
                // Non-packed dim — double the last one.
                let value = readVarint(data, &offset)
                dimsSeenNonPacked += 1
                let isLast = (dimsSeenNonPacked == nonPackedDimCount)
                result += tagBytes
                result += encodeVarint(isLast ? value * 2 : value)

            } else if fieldNum == 1, wireType == 2, hasPackedDims {
                // Packed dims — decode all, double the last, re-encode as packed LD.
                let len    = Int(readVarint(data, &offset))
                let packed = Data(data[offset ..< min(offset + len, data.count)])
                offset += len

                var dims = [UInt64]()
                var pi   = 0
                while pi < packed.count {
                    var r: UInt64 = 0; var s = 0
                    while pi < packed.count {
                        let b = UInt64(packed[pi]); pi += 1
                        r |= (b & 0x7F) << s
                        if b & 0x80 == 0 { break }
                        s += 7
                    }
                    dims.append(r)
                }
                if !dims.isEmpty { dims[dims.count - 1] *= 2 }

                var newPacked = Data()
                for d in dims { newPacked += encodeVarint(d) }
                result += encodeLD(field: 1, value: newPacked)

            } else if fieldNum == 2, wireType == 0 {
                // data_type: uint8 (2) → uint4 (21)
                let value = readVarint(data, &offset)
                result += tagBytes
                result += encodeVarint(value == 2 ? 21 : value)

            } else {
                result += tagBytes
                result += consumeValue(data, wireType: wireType, &offset)
            }
        }

        return result
    }

    /// Extract TensorProto.name (field 8, string) from raw bytes.
    private static func parseInitializerName(_ data: Data) -> String {
        var offset = 0
        while offset < data.count {
            let tag      = readVarint(data, &offset)
            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)

            if wireType == 2 {
                let len   = Int(readVarint(data, &offset))
                let bytes = Data(data[offset ..< min(offset + len, data.count)])
                offset   += len
                if fieldNum == 8, let name = String(data: bytes, encoding: .utf8) {
                    return name
                }
            } else {
                _ = consumeValue(data, wireType: wireType, &offset)
            }
        }
        return ""
    }

    // MARK: - NodeProto: strip unsupported "bits" attribute from GatherBlockQuantized nodes

    /// ORT 1.20 does not recognise the "bits" attribute on GatherBlockQuantized;
    /// it infers bit-width from the tensor type.  Strip it so the model validates.
    private static func stripGBQBitsAttribute(_ nodeData: Data) -> Data {
        let (_, opType) = parseNodeInputsAndOpType(nodeData)
        guard opType == "GatherBlockQuantized" else { return nodeData }

        var result = Data(capacity: nodeData.count)
        var offset = 0

        while offset < nodeData.count {
            let tagStart = offset
            let tag      = readVarint(nodeData, &offset)
            guard offset <= nodeData.count else { break }

            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)
            let tagBytes = Data(nodeData[tagStart ..< offset])

            if fieldNum == 5, wireType == 2 {
                // AttributeProto — skip if name == "bits"
                let len      = Int(readVarint(nodeData, &offset))
                let attrData = Data(nodeData[offset ..< min(offset + len, nodeData.count)])
                offset += len
                if parseAttributeName(attrData) != "bits" {
                    result += encodeLD(field: 5, value: attrData)
                }
            } else {
                result += tagBytes
                result += consumeValue(nodeData, wireType: wireType, &offset)
            }
        }

        return result
    }

    /// Extract AttributeProto.name (field 1, string) from raw bytes.
    private static func parseAttributeName(_ data: Data) -> String {
        var offset = 0
        while offset < data.count {
            let tag      = readVarint(data, &offset)
            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)

            if fieldNum == 1, wireType == 2 {
                let len   = Int(readVarint(data, &offset))
                let bytes = Data(data[offset ..< min(offset + len, data.count)])
                offset   += len
                return String(data: bytes, encoding: .utf8) ?? ""
            } else {
                _ = consumeValue(data, wireType: wireType, &offset)
            }
        }
        return ""
    }

    // MARK: - NodeProto: extract input / output names (for topological sort)

    private static func parseNodeIO(_ data: Data) -> (inputs: [String], outputs: [String]) {
        var inputs  = [String]()
        var outputs = [String]()
        var offset  = 0

        while offset < data.count {
            let tag      = readVarint(data, &offset)
            let wireType = Int(tag & 0x7)
            let fieldNum = Int(tag >> 3)

            if wireType == 2 {
                let len   = Int(readVarint(data, &offset))
                let bytes = Data(data[offset ..< min(offset + len, data.count)])
                offset   += len

                if let str = String(data: bytes, encoding: .utf8) {
                    switch fieldNum {
                    case 1: inputs.append(str)
                    case 2: outputs.append(str)
                    default: break
                    }
                }
            } else {
                _ = consumeValue(data, wireType: wireType, &offset)
            }
        }

        return (inputs, outputs)
    }

    // MARK: - Topological Sort (Kahn's BFS)

    private static func topologicalSort(_ nodes: [Data]) -> [Data] {
        let n  = nodes.count
        let io = nodes.map { parseNodeIO($0) }

        var producer = [String: Int]()
        for (i, node) in io.enumerated() {
            for out in node.outputs where !out.isEmpty {
                producer[out] = i
            }
        }

        var inDegree   = [Int](repeating: 0, count: n)
        var successors = [[Int]](repeating: [], count: n)

        for i in 0 ..< n {
            var seen = Set<Int>()
            for inp in io[i].inputs where !inp.isEmpty {
                guard let p = producer[inp], p != i, !seen.contains(p) else { continue }
                inDegree[i] += 1
                successors[p].append(i)
                seen.insert(p)
            }
        }

        var queue  = (0 ..< n).filter { inDegree[$0] == 0 }
        var result = [Int]()
        result.reserveCapacity(n)
        var qi = 0

        while qi < queue.count {
            let cur = queue[qi]; qi += 1
            result.append(cur)
            for nxt in successors[cur] {
                inDegree[nxt] -= 1
                if inDegree[nxt] == 0 { queue.append(nxt) }
            }
        }

        if result.count < n {
            let inResult = Set(result)
            for i in 0 ..< n where !inResult.contains(i) { result.append(i) }
        }

        return result.map { nodes[$0] }
    }
}
