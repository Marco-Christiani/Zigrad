//! Native registration of TVM CUDA tensor intrinsics.

const std = @import("std");
const tir = @import("../c/tvm/tir.zig");
const ms = @import("../c/tvm/meta_schedule.zig");

const log = std.log.scoped(.@"zg/tvm_cuda_intrinsics");

var mutex: std.Io.Mutex = .init;
var registered: bool = false;

/// Load and register the CUDA TensorIntrin bundle once per process.
pub fn ensure_registered(
    /// I/O context used to read the bundle.
    io: std.Io,
    /// Allocator used for the serialized bundle and TVM calls.
    allocator: std.mem.Allocator,
    /// Bundle generated from the same TVM revision as the compiler library.
    path: []const u8,
) !void {
    try mutex.lock(io);
    defer mutex.unlock(io);
    if (registered) return;

    const bytes = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        allocator,
        .limited(32 * 1024 * 1024),
    );
    defer allocator.free(bytes);
    const json = try allocator.dupeZ(u8, bytes);
    defer allocator.free(json);

    const intrinsic_count = tir.register_tensor_intrin_bundle(allocator, json) catch |err| switch (err) {
        error.InvalidTensorIntrinsicBundle => return error.InvalidCudaIntrinsicBundle,
        else => return err,
    };

    registered = true;
    log.info("registered {d} CUDA tensor intrinsics", .{intrinsic_count});
}

test ensure_registered {
    const config = @import("config.zig");
    const runtime = @import("runtime.zig");

    const path = std.process.Environ.getAlloc(
        std.testing.environ,
        std.testing.allocator,
        config.cuda_intrinsics_path_env,
    ) catch |err| switch (err) {
        error.EnvironmentVariableMissing => return error.SkipZigTest,
        else => return err,
    };
    defer std.testing.allocator.free(path);
    try runtime.configure(.{ .surface = .compiler });
    try runtime.ensure_loaded(.compiler);
    try ensure_registered(std.testing.io, std.testing.allocator, path);

    try std.testing.expect(try tir.tensor_intrin_registered(
        std.testing.allocator,
        "mma_sync_m16n8k8_f16f16f32",
    ));
    try std.testing.expect(try ms.default_cuda_tensor_core_rule_count(
        std.testing.allocator,
    ) > 0);
}
