//! Benchmarking. See [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs).
//!
//! # ADR
//!
//!   - Only have f32 as benchmark.rs only runs f32 hardcoded.
//!   - It is trivial to use comptime generics in zig but ideally we have something to compare to,
//!   so if more types are ever added to the saftensors benchmarks then lets update this.
const std = @import("std");
const stz = @import("root.zig");

/// Benchmark helper to create a 2MB tensor
fn getSampleData(allocator: std.mem.Allocator) !struct {
    data: []align(8) u8,
    shape: []usize,
    dtype: stz.Dtype,
} {
    // See comments regarding f32
    // 1000 x 500 elements = 500,000 elements; F32 = 4 bytes each => 2,000,000 bytes (~2 MB)
    const shape = try allocator.dupe(usize, &[_]usize{ 1000, 500 });
    const dtype = stz.Dtype.f32;
    const n = shape[0] * shape[1] * dtype.size();
    const data = try allocator.alignedAlloc(u8, .@"8", n);
    @memset(data, 0);
    return .{ .data = data, .shape = shape, .dtype = dtype };
}

const BmResult = struct {
    const Self = @This();
    total_mb: usize,
    avg: f64,
    min: f64,
    max: f64,

    fn _ns_to_us(x: f64) f64 {
        return x / @as(f64, std.time.ns_per_us);
    }

    pub fn as_us(self: Self) Self {
        return Self{
            .total_mb = self.total_mb,
            .avg = _ns_to_us(self.avg),
            .min = _ns_to_us(self.min),
            .max = _ns_to_us(self.max),
        };
    }

    pub fn print_to_writer(self: Self, writer: anytype) !void {
        try writer.print("{}_MB [min={d:.3}, avg={d:.3}, max={d:.3}]", .{ self.total_mb, self.min, self.avg, self.max });
    }
};

/// Benchmark serialization: builds 5 tensors (total ~10 MB) and measures the time to serialize.
/// Should match the official [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs)
fn benchSerialize(io: std.Io, allocator: std.mem.Allocator) !BmResult {
    const sample = try getSampleData(allocator);
    defer allocator.free(sample.data);
    defer allocator.free(sample.shape);

    const n_layers = 5;
    var total_mb: usize = 0;
    var tensorList = std.ArrayList(stz.Tensor).empty;
    defer tensorList.deinit(allocator);

    inline for (0..n_layers) |i| {
        const name = std.fmt.comptimePrint("weight{}", .{i});
        try tensorList.append(allocator, stz.Tensor{
            .name = name,
            .dtype = sample.dtype,
            .shape = sample.shape,
            .data = sample.data,
        });
        // See comments regarding f32
        total_mb += sample.data.len / @as(usize, 1e6);
    }
    // Each tensor is 2MB
    std.debug.assert(total_mb == n_layers * 2);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    // warmup
    for (0..5) |_| {
        _ = try stz.serialize_tensors(tensorList, arena.allocator());
        _ = arena.reset(.retain_capacity);
    }

    var total: u64 = 0;
    var min: u64 = std.math.maxInt(u63);
    var max: u64 = 0;
    const n = 100;
    for (0..n) |_| {
        const start = std.Io.Timestamp.now(io, .awake);
        _ = try stz.serialize_tensors(tensorList, arena.allocator());
        const elapsed = start.untilNow(io, .awake);
        _ = arena.reset(.retain_capacity);
        const time: u64 = @intCast(elapsed.toNanoseconds());
        total += time;
        if (time < min) min = time;
        if (time > max) max = time;
    }

    return BmResult{
        .total_mb = total_mb,
        .avg = @as(f64, @floatFromInt(total)) / n,
        .min = @floatFromInt(min),
        .max = @floatFromInt(max),
    };
}

/// Benchmark deserialization: takes a serialized buffer (from 5 tensors) and measures the time to load.
/// Should match the official [benchmark.rs](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/benches/benchmark.rs)
fn benchDeserialize(io: std.Io, allocator: std.mem.Allocator) !BmResult {
    const sample = try getSampleData(allocator);
    defer allocator.free(sample.data);
    defer allocator.free(sample.shape);

    const n_layers = 5;
    var total_mb: usize = 0;
    var tensorList = std.ArrayList(stz.Tensor).empty;
    defer tensorList.deinit(allocator);
    inline for (0..n_layers) |i| {
        const name = std.fmt.comptimePrint("weight{}", .{i});
        try tensorList.append(allocator, stz.Tensor{
            .name = name,
            .dtype = sample.dtype,
            .shape = sample.shape,
            .data = sample.data,
        });
        // See comments regarding f32
        total_mb += sample.data.len / @as(usize, 1e6);
    }
    // Each tensor is 2MB
    std.debug.assert(total_mb == n_layers * 2);

    const serialized = try stz.serialize_tensors(tensorList, allocator);
    defer allocator.free(serialized);

    // Just need some space to parse the header (can be up to `stz.MAX_HEADER_SIZE`)
    var buf: [1_000_000]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buf);

    // warmup
    var sum_total: usize = 0;
    for (0..5) |_| {
        var result = try stz.SafeTensorsFile.deserialize(serialized, fba.allocator());
        result.deinit();
        defer fba.reset();
        for (result.tensors) |tensor| sum_total += blk: {
            var s: usize = 1;
            for (tensor.shape) |d| s *= d;
            break :blk s;
        };
    }

    var total: u64 = 0;
    var min: u64 = std.math.maxInt(u63);
    var max: u64 = 0;
    const n = 100;
    for (0..n) |_| {
        const start = std.Io.Timestamp.now(io, .awake);
        var result = try stz.SafeTensorsFile.deserialize(serialized, fba.allocator());
        const elapsed = start.untilNow(io, .awake);
        result.deinit();
        fba.reset();
        const time: u64 = @intCast(elapsed.toNanoseconds());
        total += time;
        if (time < min) min = time;
        if (time > max) max = time;

        for (result.tensors) |tensor| sum_total += blk: {
            var s: usize = 1;
            for (tensor.shape) |d| s *= d;
            break :blk s;
        };
    }

    std.debug.print("total len: {}\n", .{sum_total});

    return BmResult{
        .total_mb = total_mb,
        .avg = @as(f64, @floatFromInt(total)) / n,
        .min = @floatFromInt(min),
        .max = @floatFromInt(max),
    };
}

/// Benchmark runner.
pub fn main(init: std.process.Init) !void {
    const ser_result = try benchSerialize(init.io, std.heap.c_allocator);
    const dser_result = try benchDeserialize(init.io, std.heap.c_allocator);

    var stdout_buffer: [1024]u8 = undefined;
    var stdout_writer = std.Io.File.stdout().writer(init.io, &stdout_buffer);
    const stdout = &stdout_writer.interface;

    try stdout.print("Serialization: ", .{});
    try ser_result.as_us().print_to_writer(stdout);
    try stdout.print("\nDeserialization: ", .{});
    try dser_result.as_us().print_to_writer(stdout);
    try stdout.print("\n", .{});
    try stdout.flush();
}
