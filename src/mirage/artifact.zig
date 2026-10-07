//! Mirage kernel artifact binary format.
//!
//! Encodes everything the dispatch layer needs to compile and launch
//!  a Mirage kernel: filtered CUDA source, workspace size, and per-kernel
//!  launch metadata with argument mappings. Integers use little-endian
//!  encoding.
//!
//! Layout:
//!   header (16 bytes)
//!   source (source_len bytes, already filtered for NVRTC)
//!   kernel descriptors (variable)

const std = @import("std");

const log = std.log.scoped(.@"zg/mirage_artifact");

/// Failures produced while serializing a Mirage artifact.
pub const EncodeError = std.mem.Allocator.Error || std.Io.Writer.Error;

/// Failures produced while validating and decoding a Mirage artifact.
pub const DecodeError = std.mem.Allocator.Error || error{InvalidArtifact};

pub const magic: u32 = 0x4D495241; // "MIRA"
pub const version: u32 = 1;

/// Describes where a kernel argument pointer comes from at dispatch time.
pub const ArgSource = enum(u32) {
    /// input_tensors[index] from the dispatch call.
    input = 0,
    /// output_tensors[index] from the dispatch call.
    output = 1,
    /// (char*)workspace + byte_offset.
    buf = 2,
};

pub const KernelArg = struct {
    source: ArgSource,
    index_or_offset: u64,
};

pub const KernelDesc = struct {
    func_name: []const u8,
    smem_bytes: u32,
    grid_dim: [3]u32,
    block_dim: [3]u32,
    args: []const KernelArg,
};

pub const Artifact = struct {
    source: []const u8,
    buf_size: u32,
    kernels: []const KernelDesc,
};

/// Serialize an artifact to an allocated byte buffer.
///
/// The caller frees the returned bytes with `allocator`.
pub fn encode(allocator: std.mem.Allocator, artifact: Artifact) EncodeError![]const u8 {
    var output: std.Io.Writer.Allocating = .init(allocator);
    errdefer output.deinit();
    const writer = &output.writer;

    // Header
    try writer.writeInt(u32, magic, .little);
    try writer.writeInt(u32, version, .little);
    try writer.writeInt(u32, @intCast(artifact.source.len), .little);
    try writer.writeInt(u32, artifact.buf_size, .little);

    // Source
    try writer.writeAll(artifact.source);

    // Number of kernels
    try writer.writeInt(u32, @intCast(artifact.kernels.len), .little);

    // Kernel descriptors
    for (artifact.kernels) |k| {
        try writer.writeInt(u32, @intCast(k.func_name.len), .little);
        try writer.writeAll(k.func_name);
        try writer.writeInt(u32, k.smem_bytes, .little);
        for (k.grid_dim) |d| try writer.writeInt(u32, d, .little);
        for (k.block_dim) |d| try writer.writeInt(u32, d, .little);
        try writer.writeInt(u32, @intCast(k.args.len), .little);
        for (k.args) |arg| {
            try writer.writeInt(u32, @backingInt(arg.source), .little);
            try writer.writeInt(u64, arg.index_or_offset, .little);
        }
    }

    return try output.toOwnedSlice();
}

/// Decode an artifact from a byte buffer.
///
/// Source and name slices point into `data`. The caller frees kernel and
///  argument slices with `allocator`.
pub fn decode(allocator: std.mem.Allocator, data: []const u8) DecodeError!Artifact {
    if (data.len < 16) return error.InvalidArtifact;

    var pos: usize = 0;

    const m = try read_u32(data, &pos);
    if (m != magic) return error.InvalidArtifact;
    const v = try read_u32(data, &pos);
    if (v != version) return error.InvalidArtifact;
    const source_len = try read_u32(data, &pos);
    const buf_size = try read_u32(data, &pos);

    if (pos + source_len > data.len) return error.InvalidArtifact;
    const source = data[pos .. pos + source_len];
    pos += source_len;

    const num_kernels = try read_u32(data, &pos);
    const kernels = try allocator.alloc(KernelDesc, num_kernels);
    var initialized_kernels: usize = 0;
    errdefer {
        for (kernels[0..initialized_kernels]) |kernel| allocator.free(kernel.args);
        allocator.free(kernels);
    }

    for (kernels) |*k| {
        const name_len = try read_u32(data, &pos);
        if (pos + name_len > data.len) return error.InvalidArtifact;
        k.func_name = data[pos .. pos + name_len];
        pos += name_len;

        k.smem_bytes = try read_u32(data, &pos);
        for (&k.grid_dim) |*d| d.* = try read_u32(data, &pos);
        for (&k.block_dim) |*d| d.* = try read_u32(data, &pos);

        const num_args = try read_u32(data, &pos);
        const args = try allocator.alloc(KernelArg, num_args);
        errdefer allocator.free(args);
        for (args) |*arg| {
            const src_val = try read_u32(data, &pos);
            arg.source = std.enums.fromInt(ArgSource, src_val) orelse
                return error.InvalidArtifact;
            arg.index_or_offset = try read_u64(data, &pos);
        }
        k.args = args;
        initialized_kernels += 1;
    }

    return .{
        .source = source,
        .buf_size = buf_size,
        .kernels = kernels,
    };
}

fn read_u32(data: []const u8, pos: *usize) error{InvalidArtifact}!u32 {
    if (pos.* + 4 > data.len) return error.InvalidArtifact;
    const val = std.mem.readInt(u32, data[pos.*..][0..4], .little);
    pos.* += 4;
    return val;
}

fn read_u64(data: []const u8, pos: *usize) error{InvalidArtifact}!u64 {
    if (pos.* + 8 > data.len) return error.InvalidArtifact;
    const val = std.mem.readInt(u64, data[pos.*..][0..8], .little);
    pos.* += 8;
    return val;
}

const test_args = [_]KernelArg{
    .{ .source = .input, .index_or_offset = 0 },
    .{ .source = .input, .index_or_offset = 1 },
    .{ .source = .output, .index_or_offset = 0 },
};

const test_kernels = [_]KernelDesc{.{
    .func_name = "custom_kernel_0",
    .smem_bytes = 49152,
    .grid_dim = .{ 1, 1, 1 },
    .block_dim = .{ 128, 1, 1 },
    .args = &test_args,
}};

const test_artifact: Artifact = .{
    .source = "__global__ void custom_kernel_0() {}",
    .buf_size = 4096,
    .kernels = &test_kernels,
};

test "encode-decode round trip" {
    const allocator = std.testing.allocator;

    const encoded = try encode(allocator, test_artifact);
    defer allocator.free(encoded);

    const decoded = try decode(allocator, encoded);
    defer allocator.free(decoded.kernels);
    defer allocator.free(decoded.kernels[0].args);

    try std.testing.expectEqualStrings(test_artifact.source, decoded.source);
    try std.testing.expectEqual(test_artifact.buf_size, decoded.buf_size);
    try std.testing.expectEqual(test_artifact.kernels.len, decoded.kernels.len);
    try std.testing.expectEqualStrings("custom_kernel_0", decoded.kernels[0].func_name);
    try std.testing.expectEqual(@as(u32, 49152), decoded.kernels[0].smem_bytes);
    try std.testing.expectEqual(@as(usize, 3), decoded.kernels[0].args.len);
    try std.testing.expectEqual(ArgSource.input, decoded.kernels[0].args[0].source);
    try std.testing.expectEqual(ArgSource.output, decoded.kernels[0].args[2].source);
}

test "decode rejects truncated artifacts" {
    const allocator = std.testing.allocator;
    const encoded = try encode(allocator, test_artifact);
    defer allocator.free(encoded);

    for (0..encoded.len) |len| {
        try std.testing.expectError(error.InvalidArtifact, decode(allocator, encoded[0..len]));
    }
}

test "decode rejects invalid argument sources" {
    const allocator = std.testing.allocator;
    const encoded = try encode(allocator, test_artifact);
    defer allocator.free(encoded);

    const corrupted = try allocator.dupe(u8, encoded);
    defer allocator.free(corrupted);

    const fixed_u32_fields = 4 + 1 + 1 + 1 + 3 + 3 + 1;
    const first_arg_source = @sizeOf(u32) * fixed_u32_fields +
        test_artifact.source.len + test_kernels[0].func_name.len;
    std.mem.writeInt(u32, corrupted[first_arg_source..][0..4], 99, .little);
    try std.testing.expectError(error.InvalidArtifact, decode(allocator, corrupted));
}
