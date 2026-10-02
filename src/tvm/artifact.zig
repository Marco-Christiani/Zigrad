//! Storage and loading for tuned TVM modules.

const std = @import("std");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const RuntimeModule = tvm_runtime.RuntimeModule;
const TargetKind = @import("config.zig").TargetKind;
const Cache = @import("../cache.zig").Cache;
const DType = @import("../dtype.zig").DType;
const export_mod = @import("export.zig");
const Linker = @import("../toolchain/linker.zig").Linker;

const log = std.log.scoped(.@"zg/tvm_artifact");

/// Loaded TVM runtime module for one cached artifact.
pub const LoadedArtifact = struct {
    module: RuntimeModule,
    main_func: tvm_runtime.Function,

    /// Release the loaded function and module.
    pub fn deinit(self: *LoadedArtifact) void {
        self.main_func.deinit();
        self.module.deinit();
        self.* = undefined;
    }

    /// Invoke the tuned main function with the given arguments.
    pub fn invoke(
        self: LoadedArtifact,
        allocator: std.mem.Allocator,
        tensors: []const tvm_runtime.Tensor,
    ) !void {
        try self.main_func.call(allocator, tensors);
    }
};

const cache_key_format_version: u32 = 1;

/// Lowercase hexadecimal BLAKE3 cache key.
pub const CacheKey = struct {
    buf: [std.crypto.hash.Blake3.digest_length * 2]u8,

    /// Return the key bytes.
    pub fn slice(self: *const CacheKey) []const u8 {
        return &self.buf;
    }
};

/// Compute the cache key for a matmul signature and target.
pub fn matmul_cache_key(
    /// Cache fingerprint for compiler inputs that affect generated artifacts.
    compiler_fingerprint: []const u8,
    target: []const u8,
    dtype: DType,
    m: i64,
    n: i64,
    k: i64,
) CacheKey {
    var hashing = std.crypto.hash.Blake3.init(.{});
    hashing.update("zigrad.tvm.matmul");
    hash_int(&hashing, u32, cache_key_format_version);
    hash_bytes(&hashing, compiler_fingerprint);
    hash_bytes(&hashing, target);
    hash_bytes(&hashing, dtype.name());
    hash_int(&hashing, i64, m);
    hash_int(&hashing, i64, n);
    hash_int(&hashing, i64, k);

    var digest: [std.crypto.hash.Blake3.digest_length]u8 = undefined;
    hashing.final(&digest);
    return .{ .buf = std.fmt.bytesToHex(digest, .lower) };
}

fn hash_bytes(hashing: *std.crypto.hash.Blake3, bytes: []const u8) void {
    hash_int(hashing, u64, @intCast(bytes.len));
    hashing.update(bytes);
}

fn hash_int(hashing: *std.crypto.hash.Blake3, comptime Int: type, value: Int) void {
    var encoded: [@sizeOf(Int)]u8 = undefined;
    std.mem.writeInt(Int, &encoded, value, .little);
    hashing.update(&encoded);
}

test matmul_cache_key {
    const sm_80 = matmul_cache_key("compiler-a", "cuda -arch=sm_80", .f32, 128, 128, 128);
    const sm_89 = matmul_cache_key("compiler-a", "cuda -arch=sm_89", .f32, 128, 128, 128);
    const float16 = matmul_cache_key("compiler-a", "cuda -arch=sm_89", .f16, 128, 128, 128);
    const compiler_b = matmul_cache_key("compiler-b", "cuda -arch=sm_89", .f32, 128, 128, 128);
    try std.testing.expect(!std.mem.eql(u8, sm_80.slice(), sm_89.slice()));
    try std.testing.expect(!std.mem.eql(u8, float16.slice(), sm_89.slice()));
    try std.testing.expect(!std.mem.eql(u8, compiler_b.slice(), sm_89.slice()));
}

const artifact_name = "kernel.so";
const pending_artifact_name = "kernel.pending.so";

/// Export and atomically publish a compiled module in `work_cache`.
pub fn publish(
    /// Compiled module to export.
    module: RuntimeModule,
    /// I/O context used to export and publish the module.
    io: std.Io,
    /// Allocator used by TVM and the linker.
    allocator: std.mem.Allocator,
    /// Cache directory dedicated to one workload and target.
    work_cache: *const Cache,
    /// Selects host-only or CUDA host-device export.
    target: TargetKind,
    /// Linker used to produce the ELF shared library.
    linker: Linker,
) !void {
    var pending = try work_cache.join(pending_artifact_name);
    const pending_path = pending.pathZ();
    const published = try work_cache.join(artifact_name);
    const cwd = std.Io.Dir.cwd();
    defer cwd.deleteFile(io, pending.path()) catch |err| switch (err) {
        error.FileNotFound => {},
        else => log.warn(
            "failed to remove pending artifact: {s}",
            .{@errorName(err)},
        ),
    };

    try export_mod.export_shared(
        module,
        io,
        allocator,
        pending_path,
        target,
        linker,
    );
    try cwd.rename(pending.path(), cwd, published.path(), io);
}

/// Load the published module from `work_cache`, or return `null` when absent.
pub fn load_cached(
    /// I/O context used to inspect and load the artifact.
    io: std.Io,
    /// Allocator used by TVM packed calls.
    allocator: std.mem.Allocator,
    /// Cache directory dedicated to one workload and target.
    work_cache: *const Cache,
) !?LoadedArtifact {
    var path = try work_cache.join(artifact_name);
    std.Io.Dir.cwd().access(io, path.path(), .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    const path_z = path.pathZ();
    var module = try RuntimeModule.load_from_file(allocator, path_z);
    errdefer module.deinit();
    const main_func = try module.get_function(allocator, "main", true);
    return .{ .module = module, .main_func = main_func };
}

/// Read the cached artifact bytes for one workload and target.
pub fn read_cached_bytes(
    /// I/O context used to read the artifact.
    io: std.Io,
    /// Allocator used for the returned bytes.
    allocator: std.mem.Allocator,
    /// Cache directory dedicated to one workload and target.
    work_cache: *const Cache,
) !?[]u8 {
    const path = try work_cache.join(artifact_name);
    std.Io.Dir.cwd().access(io, path.path(), .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    return try std.Io.Dir.cwd().readFileAlloc(
        io,
        path.path(),
        allocator,
        .limited(100 * 1024 * 1024),
    );
}
