//! TVM kernel dispatch state.
//!
//! The state prepares and caches loaded TVM runtime modules for dispatch.
const std = @import("std");
const kernel = @import("kernel");
const dlpack = @import("../c/dlpack.zig");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const integration_runtime = @import("runtime.zig");
const Cache = @import("../cache.zig").Cache;
const TypedPtr = @import("rtti").TypedPtr;

const log = std.log.scoped(.@"zg/tvm_dispatch");

const TvmDispatchEntry = struct {
    module: tvm_runtime.RuntimeModule,
    main_func: tvm_runtime.Function,

    fn deinit(self: *TvmDispatchEntry) void {
        self.main_func.deinit();
        self.module.deinit();
        self.* = undefined;
    }
};

/// Caches TVM runtime modules used for kernel dispatch.
///
/// A single instance is shared by its provider's artifacts. Dispatch is
///  single-threaded until the kernel-provider contract supplies synchronization.
pub const TvmDispatchState = struct {
    io: std.Io,
    cache: std.AutoHashMap(ArtifactHash, TvmDispatchEntry),
    artifact_cache: Cache,

    /// Initialize an empty dispatch cache.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        artifact_cache: Cache,
    ) TvmDispatchState {
        return .{
            .io = io,
            .cache = .init(allocator),
            .artifact_cache = artifact_cache,
        };
    }

    /// Release every loaded dispatch entry.
    pub fn deinit(self: *TvmDispatchState) void {
        var it = self.cache.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.cache.deinit();
        self.* = undefined;
    }

    /// Prepare one compiled artifact for execution.
    pub fn prepare(
        provider_ctx: TypedPtr,
        artifact_data: []const u8,
        kernel_key: []const u8,
        _: kernel.PrepareContext,
    ) kernel.PrepareError!void {
        const self = provider_ctx.cast(TvmDispatchState);
        self.prepare_impl(artifact_data) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.TvmLoadFailed => return error.ProviderLoadFailed,
            else => {
                log.err("failed to prepare TVM artifact '{s}': {s}", .{
                    kernel_key,
                    @errorName(err),
                });
                return error.PrepareFailed;
            },
        };
    }

    fn prepare_impl(self: *TvmDispatchState, artifact_data: []const u8) !void {
        try integration_runtime.ensure_loaded(.runtime);

        const artifact_hash = hash_artifact(artifact_data);
        if (self.cache.contains(artifact_hash)) return;

        var loaded = try load_dispatch_entry(
            self.io,
            &self.artifact_cache,
            artifact_hash,
            artifact_data,
        );
        errdefer loaded.deinit();
        try self.cache.put(artifact_hash, loaded);
    }

    /// Dispatch one prepared TVM artifact.
    pub fn dispatch(
        provider_ctx: TypedPtr,
        artifact_data: []const u8,
        kernel_key: []const u8,
        ctx: kernel.DispatchContext,
    ) kernel.DispatchError!void {
        const self = provider_ctx.cast(TvmDispatchState);
        self.dispatch_impl(artifact_data, kernel_key, ctx) catch |err| {
            log.err("tvm dispatch failed for '{s}': {s}", .{ kernel_key, @errorName(err) });
            return error.DispatchFailed;
        };
    }

    fn dispatch_impl(
        self: *TvmDispatchState,
        artifact_data: []const u8,
        _: []const u8,
        ctx: kernel.DispatchContext,
    ) !void {
        const artifact_hash = hash_artifact(artifact_data);
        const entry = self.cache.get(artifact_hash) orelse
            return error.ArtifactNotPrepared;

        if (ctx.device.platform.eql(.cuda)) {
            if (ctx.stream) |stream_ptr| {
                try configure_cuda_stream(stream_ptr, ctx.device.ordinal);
            }
        }

        const device_type: dlpack.DeviceType = if (ctx.device.platform.eql(.cuda))
            .cuda
        else if (ctx.device.platform.eql(.cpu))
            .cpu
        else
            return error.UnsupportedDevice;

        var tensors: [16]tvm_runtime.Tensor = undefined;
        const total = ctx.inputs.len + ctx.outputs.len;
        if (total > tensors.len) return error.TooManyArguments;

        var initialized: usize = 0;
        defer for (tensors[0..initialized]) |*tensor| tensor.deinit();

        for (ctx.inputs, 0..) |buf, i| {
            tensors[i] = try tensor_from_buffer_desc(buf, device_type, ctx.device.ordinal);
            initialized += 1;
        }
        for (ctx.outputs, 0..) |buf, i| {
            const idx = ctx.inputs.len + i;
            tensors[idx] = try tensor_from_buffer_desc(buf, device_type, ctx.device.ordinal);
            initialized += 1;
        }

        try entry.main_func.call(std.heap.c_allocator, tensors[0..total]);
    }
};

fn tensor_from_buffer_desc(buf: kernel.BufferDesc, device_type: dlpack.DeviceType, device_id: i32) !tvm_runtime.Tensor {
    const dl_tensor: dlpack.Tensor = .{
        .data = buf.data,
        .device = .{ .device_type = device_type, .device_id = device_id },
        .ndim = @intCast(buf.rank),
        .dtype = kernel_dtype_to_dlpack(buf.dtype),
        .shape = @constCast(buf.dims.ptr),
        .strides = null,
        .byte_offset = 0,
    };
    const managed = try dlpack.ManagedTensor.heap_borrowing(dl_tensor);
    errdefer managed.deleter.?(managed);
    return try tvm_runtime.Tensor.from_dlpack(managed);
}

fn kernel_dtype_to_dlpack(dtype: kernel.DType) dlpack.DataType {
    return switch (dtype) {
        .f16 => .{ .code = .float, .bits = 16, .lanes = 1 },
        .bf16 => .{ .code = .bfloat, .bits = 16, .lanes = 1 },
        .f32 => .{ .code = .float, .bits = 32, .lanes = 1 },
        .f64 => .{ .code = .float, .bits = 64, .lanes = 1 },
        .i8 => .{ .code = .int, .bits = 8, .lanes = 1 },
        .u8 => .{ .code = .uint, .bits = 8, .lanes = 1 },
        .i32 => .{ .code = .int, .bits = 32, .lanes = 1 },
        .i64 => .{ .code = .int, .bits = 64, .lanes = 1 },
        .u32 => .{ .code = .uint, .bits = 32, .lanes = 1 },
        .u64 => .{ .code = .uint, .bits = 64, .lanes = 1 },
        .bool => .{ .code = .uint, .bits = 8, .lanes = 1 },
    };
}

fn configure_cuda_stream(stream_ptr: *anyopaque, device_id: i32) !void {
    try tvm_runtime.set_stream(
        std.heap.c_allocator,
        .cuda,
        device_id,
        stream_ptr,
    );
}

fn load_dispatch_entry(
    io: std.Io,
    artifact_cache: *const Cache,
    artifact_hash: ArtifactHash,
    artifact_data: []const u8,
) !TvmDispatchEntry {
    var name_buf: [128]u8 = undefined;
    const digest_hex = std.fmt.bytesToHex(artifact_hash, .lower);
    const filename = try std.fmt.bufPrint(&name_buf, "{s}.so", .{&digest_hex});
    const tvm_cache = try artifact_cache.subdir(io, "tvm", .{});
    var dispatch_cache = try tvm_cache.subdir(io, "dispatch", .{});
    var resolved = try dispatch_cache.join(filename);
    const path = resolved.pathZ();

    {
        var file = try std.Io.Dir.cwd().createFile(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, artifact_data);
    }

    var module = try tvm_runtime.RuntimeModule.load_from_file(std.heap.c_allocator, path);
    errdefer module.deinit();

    const main_func = try module.get_function(std.heap.c_allocator, "main", true);
    return .{
        .module = module,
        .main_func = main_func,
    };
}

const ArtifactHash = [std.crypto.hash.Blake3.digest_length]u8;

fn hash_artifact(artifact_data: []const u8) ArtifactHash {
    var digest: ArtifactHash = undefined;
    std.crypto.hash.Blake3.hash(artifact_data, &digest, .{});
    return digest;
}
