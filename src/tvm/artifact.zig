//! Storage and loading for tuned TVM modules.

const std = @import("std");
const builtin = @import("builtin");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const RuntimeModule = tvm_runtime.RuntimeModule;
const TargetKind = @import("config.zig").TargetKind;
const Cache = @import("../cache.zig").Cache;
const DType = @import("dtype").DType;
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

const cache_key_format_version: u32 = 2;

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
    identity: @import("config.zig").BuildIdentity,
    target: []const u8,
    dtype: DType,
    m: i64,
    n: i64,
    k: i64,
) CacheKey {
    var hashing = std.crypto.hash.Blake3.init(.{});
    hashing.update("zigrad.tvm.matmul");
    hash_int(&hashing, u32, cache_key_format_version);
    hash_bytes(&hashing, &identity.digest);
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
    const sm_80 = matmul_cache_key(@import("config.zig").BuildIdentity.from_external("compiler-a"), "cuda -arch=sm_80", .f32, 128, 128, 128);
    const sm_89 = matmul_cache_key(@import("config.zig").BuildIdentity.from_external("compiler-a"), "cuda -arch=sm_89", .f32, 128, 128, 128);
    const float16 = matmul_cache_key(@import("config.zig").BuildIdentity.from_external("compiler-a"), "cuda -arch=sm_89", .f16, 128, 128, 128);
    const compiler_b = matmul_cache_key(@import("config.zig").BuildIdentity.from_external("compiler-b"), "cuda -arch=sm_89", .f32, 128, 128, 128);
    try std.testing.expect(!std.mem.eql(u8, sm_80.slice(), sm_89.slice()));
    try std.testing.expect(!std.mem.eql(u8, float16.slice(), sm_89.slice()));
    try std.testing.expect(!std.mem.eql(u8, compiler_b.slice(), sm_89.slice()));
}

pub const artifact_name = "kernel.so";

/// Immutable publication generation, created under the owning workload lock.
///  Readers retain the selected generation. Successful generations remain until cache removal.
pub const Generation = struct {
    cache: Cache,
    name: [32]u8,

    pub fn create(io: std.Io, work: *const Cache) !Generation {
        var random: [16]u8 = undefined;
        io.random(&random);
        const name = std.fmt.bytesToHex(random, .lower);
        const generations = try work.subdir(io, "generations", .{});
        const cache = try generations.join(&name);
        try std.Io.Dir.cwd().createDir(io, cache.path(), .default_dir);
        return .{ .cache = cache, .name = name };
    }

    /// Export the measured file without changing the current generation.
    pub fn stage(self: *const Generation, module: RuntimeModule, io: std.Io, allocator: std.mem.Allocator, target: TargetKind, linker: Linker) !void {
        var path = try self.cache.join(artifact_name);
        try export_mod.export_shared(module, io, allocator, path.pathZ(), target, linker);
    }

    /// Write complete evidence before atomically replacing the current pointer.
    pub fn publish(self: *const Generation, io: std.Io, allocator: std.mem.Allocator, work: *const Cache, evidence: anytype) !void {
        const data = try std.json.Stringify.valueAlloc(allocator, evidence, .{});
        defer allocator.free(data);
        const record = try self.cache.join("evidence.json");
        const cwd = std.Io.Dir.cwd();
        try cwd.writeFile(io, .{ .sub_path = record.path(), .data = data });
        const pending = try work.join("current.pending");
        const current = try work.join("current");
        try cwd.writeFile(io, .{ .sub_path = pending.path(), .data = &self.name });
        try cwd.rename(pending.path(), cwd, current.path(), io);
    }

    pub fn digest(self: *const Generation, io: std.Io, allocator: std.mem.Allocator) ![64]u8 {
        const path = try self.cache.join(artifact_name);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path.path(), allocator, .limited(100 * 1024 * 1024));
        defer allocator.free(bytes);
        var hash: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &hash, .{});
        return std.fmt.bytesToHex(hash, .lower);
    }
};

fn current_generation(io: std.Io, allocator: std.mem.Allocator, work: *const Cache) !?Cache {
    const pointer = try work.join("current");
    const name = std.Io.Dir.cwd().readFileAlloc(io, pointer.path(), allocator, .limited(33)) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer allocator.free(name);
    if (name.len != 32) {
        if (!builtin.is_test) log.err("current generation pointer in {s} has invalid length {d}", .{ work.path(), name.len });
        return error.InvalidGeneration;
    }
    for (name) |ch| if (!std.ascii.isHex(ch)) {
        if (!builtin.is_test) log.err("current generation pointer in {s} contains a non-hexadecimal byte", .{work.path()});
        return error.InvalidGeneration;
    };
    const generations = try work.join("generations");
    return try generations.join(name);
}

/// Stable database partition for every timing setting.
pub fn protocol_key(protocol: tvm_runtime.TimeEvaluatorOptions) CacheKey {
    var hashing = std.crypto.hash.Blake3.init(.{});
    hashing.update("zigrad.tvm.protocol.v1");
    hash_int(&hashing, i32, protocol.number);
    hash_int(&hashing, i32, protocol.repeats);
    hash_int(&hashing, i32, protocol.min_repeat_ms);
    hash_int(&hashing, i32, protocol.cache_flush_bytes);
    var digest: [32]u8 = undefined;
    hashing.final(&digest);
    return .{ .buf = std.fmt.bytesToHex(digest, .lower) };
}

test "protocol_key distinguishes each timing setting" {
    const base = protocol_key(.{});
    inline for (std.meta.fields(tvm_runtime.TimeEvaluatorOptions)) |field| {
        var protocol: tvm_runtime.TimeEvaluatorOptions = .{};
        @field(protocol, field.name) += 1;
        const changed = protocol_key(protocol);
        try std.testing.expect(!std.mem.eql(u8, base.slice(), changed.slice()));
    }
    const same = protocol_key(.{});
    try std.testing.expectEqualStrings(base.slice(), same.slice());
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
    const generation = try current_generation(io, allocator, work_cache) orelse return null;
    var path = try generation.join(artifact_name);
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
    const generation = try current_generation(io, allocator, work_cache) orelse return null;
    const path = try generation.join(artifact_name);
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

test "Generation keeps readers coherent and failed staging preserves current" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var environ: std.process.Environ.Map = .init(allocator);
    defer environ.deinit();
    const work = try Cache.init(io, &environ, .{ .root = root });
    const first = try Generation.create(io, &work);
    const first_file = try first.cache.join(artifact_name);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = first_file.path(), .data = "first artifact" });
    try first.publish(io, allocator, &work, .{ .artifact_digest = try first.digest(io, allocator), .protocol = tvm_runtime.TimeEvaluatorOptions{} });
    const reader = (try current_generation(io, allocator, &work)).?;
    const unpublished = try Generation.create(io, &work);
    const failed_file = try unpublished.cache.join(artifact_name);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = failed_file.path(), .data = "unmeasured artifact" });
    const unchanged = (try current_generation(io, allocator, &work)).?;
    try std.testing.expectEqualStrings(reader.path(), unchanged.path());
    try unpublished.publish(io, allocator, &work, .{ .artifact_digest = try unpublished.digest(io, allocator), .protocol = tvm_runtime.TimeEvaluatorOptions{} });
    const changed = (try current_generation(io, allocator, &work)).?;
    try std.testing.expect(!std.mem.eql(u8, changed.path(), reader.path()));
    const old_file = try reader.join(artifact_name);
    const old_bytes = try std.Io.Dir.cwd().readFileAlloc(io, old_file.path(), allocator, .limited(100));
    defer allocator.free(old_bytes);
    try std.testing.expectEqualStrings("first artifact", old_bytes);
    const current_bytes = (try read_cached_bytes(io, allocator, &work)).?;
    defer allocator.free(current_bytes);
    try std.testing.expectEqualStrings("unmeasured artifact", current_bytes);
}
