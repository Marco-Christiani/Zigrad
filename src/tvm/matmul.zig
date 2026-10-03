//! TVM matmul tuning and execution.

const std = @import("std");
const builtin = @import("builtin");

const device_mod = @import("device");
const DType = @import("dtype").DType;
const dlpack = @import("../c/dlpack.zig");
const ffi = @import("../c/tvm/ffi.zig");
const tvm_compile = @import("../c/tvm/compile.zig");
const tvm_runtime = @import("../c/tvm/runtime.zig");
const ms = @import("../c/tvm/meta_schedule.zig");
const tir = @import("../c/tvm/tir.zig");
const te = @import("../c/tvm/te.zig");
const Cache = @import("../cache.zig").Cache;
const artifact = @import("artifact.zig");
const config = @import("config.zig");
pub const TargetKind = config.TargetKind;
const integration_runtime = @import("runtime.zig");
const tune_mod = @import("tune.zig");

const log = std.log.scoped(.@"zg/tvm_matmul");

/// Elements admitted for matmul. Both members multiply and reduce in f32.
///  f16 converts the completed reduction once to f16. f32 produces f32 output.
pub const ElementType = enum {
    f16,
    f32,

    pub fn from_dtype(value: DType) ?ElementType {
        return std.meta.stringToEnum(ElementType, @tagName(value));
    }

    pub fn dtype(self: ElementType) DType {
        return switch (self) {
            .f16 => .f16,
            .f32 => .f32,
        };
    }

    pub fn Type(comptime self: ElementType) type {
        return switch (self) {
            .f16 => f16,
            .f32 => f32,
        };
    }

    fn matmul_options(self: ElementType) te.MatmulOptions {
        return switch (self) {
            .f16 => .{ .accumulation_dtype = .f32, .output_dtype = .f16 },
            .f32 => .{ .accumulation_dtype = .f32, .output_dtype = .f32 },
        };
    }
};

test "ElementType admission and numerical policy" {
    inline for (std.meta.tags(DType)) |dtype| {
        const admitted = ElementType.from_dtype(dtype);
        try std.testing.expectEqual(dtype == .f16 or dtype == .f32, admitted != null);
        if (admitted) |element| {
            try std.testing.expectEqual(dtype, element.dtype());
            try std.testing.expectEqual(.f32, element.matmul_options().accumulation_dtype);
            try std.testing.expectEqual(dtype, element.matmul_options().output_dtype);
        }
    }
}

/// Matrix-multiplication signature supported by the TVM provider.
pub const Shape = struct {
    /// Left-hand matrix row count.
    m: i64,
    /// Right-hand matrix column count.
    n: i64,
    /// Contracting dimension.
    k: i64,
    /// Shared input and output element type.
    dtype: ElementType = .f32,
};

/// Options for tuning one matrix multiplication shape.
pub const TuneOptions = struct {
    /// Target-specific compiler inputs resolved by application composition.
    compile: config.CompileConfig,

    /// Device used for target detection and candidate measurement.
    device: device_mod.Device,

    /// Clear the current build/protocol database before tuning.
    retune: bool = false,

    /// Maximum measured candidates.
    max_trials: u32 = 64,

    /// Candidates submitted per tuning iteration.
    trials_per_iter: u32 = 16,

    /// Repetition and cache policy for TVM schedule measurements.
    measurement: tvm_runtime.TimeEvaluatorOptions = .{},
};

/// Selected result from one tuning run.
pub const TuneResult = struct {
    /// Mean timing of the final measured publication file.
    best_time_us: f64,
    /// Evaluator repeats for the final measured publication file.
    sample_count: usize,
    /// File name inside the measured immutable generation.
    artifact_name: []const u8,
    /// SHA-256 of the measured shared library.
    artifact_digest: [64]u8,
    /// Immutable generation containing the measured file and evidence.
    generation: [32]u8,
    /// Protocol used to measure the final file.
    protocol: tvm_runtime.TimeEvaluatorOptions,
};

/// Tune one matrix multiplication and publish the selected artifact.
pub fn tune(
    /// I/O context used for cache files and candidate measurements.
    io: std.Io,
    /// Allocator used by TVM calls and temporary tuning state.
    allocator: std.mem.Allocator,
    /// Root for artifacts and tuning records.
    artifact_cache: *const Cache,
    /// Matrix dimensions and element type.
    shape: Shape,
    /// Compiler inputs, device, and tuning limits.
    options: TuneOptions,
) !TuneResult {
    try integration_runtime.ensure_loaded(.compiler);

    const target = options.compile.target.kind();
    if (!target.accepts(options.device)) return error.TargetDeviceMismatch;
    var tvm_target = try tir.Target.resolve(
        allocator,
        options.compile.target,
        options.device,
    );
    defer tvm_target.deinit();
    const location = try cache_location(
        io,
        artifact_cache,
        shape,
        options.compile.build_identity(),
        tvm_target.description(),
        true,
    );
    var tuning_lock = try acquire_tuning_lock(io, &location.lock);
    defer tuning_lock.deinit();

    try options.measurement.validate();
    const measurements = try location.work.subdir(io, "measurements", .{});
    const protocol_key = artifact.protocol_key(options.measurement);
    const database = try measurements.subdir(io, protocol_key.slice(), .{});
    if (options.retune) try reset_tuning_records(io, &database);

    var ir_module = try build_ir_module(allocator, shape, options.compile.target.kind());
    defer ir_module.deinit();
    const shape_a = [_]i64{ shape.m, shape.k };
    const shape_b = [_]i64{ shape.k, shape.n };
    const shape_c = [_]i64{ shape.m, shape.n };
    const shapes = [_][]const i64{ &shape_a, &shape_b, &shape_c };

    var tuned = try tune_mod.tune(
        io,
        allocator,
        ir_module,
        shape.dtype,
        &shapes,
        .{
            .linker = options.compile.linker,
            .resolved = &tvm_target,
            .work_cache = &database,
            .max_trials = options.max_trials,
            .trials_per_iter = options.trials_per_iter,
            .measurement = options.measurement,
        },
    );
    defer tuned.deinit();

    var compiled = try tvm_compile.lower_and_compile(
        allocator,
        &tuned,
        tvm_target.target(),
        target,
    );
    defer compiled.deinit();
    return publish_module(io, allocator, &location.work, shape, options.compile, tvm_target.description(), options.device, options.measurement, compiled);
}

fn publish_module(io: std.Io, allocator: std.mem.Allocator, work: *const Cache, shape: Shape, compile_config: config.CompileConfig, target_description: []const u8, device: device_mod.Device, protocol: tvm_runtime.TimeEvaluatorOptions, compiled: tvm_runtime.RuntimeModule) !TuneResult {
    try protocol.validate();
    const shapes = [_][]const i64{ &.{ shape.m, shape.k }, &.{ shape.k, shape.n }, &.{ shape.m, shape.n } };
    const generation = try artifact.Generation.create(io, work);
    errdefer std.Io.Dir.cwd().deleteTree(io, generation.cache.path()) catch |err| {
        log.warn("failed to remove unpublished generation: {s}", .{@errorName(err)});
    };
    try generation.stage(compiled, io, allocator, compile_config.target.kind(), compile_config.linker);
    var path = try generation.cache.join(artifact.artifact_name);
    var measured = try tvm_runtime.RuntimeModule.load_from_file(allocator, path.pathZ());
    defer measured.deinit();
    const samples = try tune_mod.measure_artifact(allocator, measured, shape.dtype, &shapes, device, compile_config.target.kind(), protocol);
    defer allocator.free(samples);
    return finish_publication(io, allocator, work, &generation, shape, compile_config, target_description, protocol, samples);
}

fn finish_publication(io: std.Io, allocator: std.mem.Allocator, work: *const Cache, generation: *const artifact.Generation, shape: Shape, compile_config: config.CompileConfig, target_description: []const u8, protocol: tvm_runtime.TimeEvaluatorOptions, samples: []const f64) !TuneResult {
    if (samples.len != protocol.repeats) {
        if (!builtin.is_test) log.err("generation {s} has {d} timing samples, expected {d}", .{ generation.name, samples.len, protocol.repeats });
        return error.InvalidTimingResult;
    }
    var sum: f64 = 0;
    for (samples) |sample| {
        if (!std.math.isFinite(sample) or sample <= 0) {
            if (!builtin.is_test) log.err("generation {s} has invalid timing sample {d}", .{ generation.name, sample });
            return error.InvalidTimingResult;
        }
        sum += sample;
    }
    const digest = try generation.digest(io, allocator);
    const identity = config.BuildIdentity.from_external(compile_config.external_identity);
    try generation.publish(io, allocator, work, .{
        .schema = 1,
        .generation = &generation.name,
        .artifact = artifact.artifact_name,
        .artifact_digest = &digest,
        .lowering_identity = @import("build_options").tvm_lowering_identity,
        .external_identity = compile_config.external_identity,
        .build_identity = &identity.digest,
        .target = target_description,
        .shape = shape,
        .protocol = protocol,
        .samples_seconds = samples,
    });
    return .{
        .best_time_us = sum / @as(f64, @floatFromInt(samples.len)) * 1e6,
        .sample_count = samples.len,
        .artifact_name = artifact.artifact_name,
        .artifact_digest = digest,
        .generation = generation.name,
        .protocol = protocol,
    };
}

/// Bytes and cache key for one published matmul artifact.
///
/// The caller frees `bytes` with the allocator passed to `load_artifact`.
pub const CachedArtifact = struct {
    /// Cache key derived from the compiler fingerprint, target, and shape.
    key: artifact.CacheKey,
    /// Shared-library bytes. Free with the allocator passed to `load_artifact`.
    bytes: []u8,
};

/// Read a published matmul artifact.
pub fn load_artifact(
    /// I/O context used to read the artifact.
    io: std.Io,
    /// Allocator used for the returned artifact bytes.
    allocator: std.mem.Allocator,
    /// Root containing TVM artifacts.
    artifact_cache: *const Cache,
    /// Matrix dimensions and element type.
    shape: Shape,
    /// Opaque identity of the lowering build and external inputs.
    compiler_identity: config.BuildIdentity,
    /// Compilation target.
    target: TargetKind,
    /// Device used to complete the target description.
    device: device_mod.Device,
) !?CachedArtifact {
    try integration_runtime.ensure_loaded(.compiler);
    if (!target.accepts(device)) return error.TargetDeviceMismatch;
    var target_description = try tir.Target.describe(
        allocator,
        target,
        device.ordinal,
    );
    defer target_description.deinit();
    const location = try cache_location(
        io,
        artifact_cache,
        shape,
        compiler_identity,
        target_description.description,
        false,
    );
    const bytes = try artifact.read_cached_bytes(
        io,
        allocator,
        &location.work,
    ) orelse return null;
    return .{ .key = location.key, .bytes = bytes };
}

const CachedMatmulState = struct {
    allocator: std.mem.Allocator,
    target: TargetKind,
    device: device_mod.Device,
    shape: Shape,
    artifact: artifact.LoadedArtifact,
};

/// Loaded matmul artifact and execution state.
pub const CachedMatmul = opaque {
    /// Load the published implementation for one shape and target.
    pub fn load(
        /// I/O context used to read the artifact.
        io: std.Io,
        /// Allocator used until `deinit`.
        allocator: std.mem.Allocator,
        /// Root containing TVM artifacts.
        artifact_cache: *const Cache,
        /// Matrix dimensions and element type.
        shape: Shape,
        /// Opaque identity of the lowering build and external inputs.
        compiler_identity: config.BuildIdentity,
        /// Compilation target.
        target: TargetKind,
        /// Device used to complete the target description and execute.
        device: device_mod.Device,
    ) !?*CachedMatmul {
        try integration_runtime.ensure_loaded(.runtime);

        if (!target.accepts(device)) return error.TargetDeviceMismatch;
        var target_description = try tir.Target.describe(
            allocator,
            target,
            device.ordinal,
        );
        defer target_description.deinit();
        const location = try cache_location(
            io,
            artifact_cache,
            shape,
            compiler_identity,
            target_description.description,
            false,
        );
        var loaded = try artifact.load_cached(
            io,
            allocator,
            &location.work,
        ) orelse return null;
        errdefer loaded.deinit();

        const state = try allocator.create(CachedMatmulState);
        state.* = .{
            .allocator = allocator,
            .target = target,
            .device = device,
            .shape = shape,
            .artifact = loaded,
        };
        return @ptrCast(state);
    }

    /// Release the loaded module and wrapper state.
    pub fn deinit(self: *CachedMatmul) void {
        const state = state_from(self);
        const allocator = state.allocator;
        state.artifact.deinit();
        allocator.destroy(state);
    }

    /// Execute with buffers matching the admitted element type.
    ///  CPU tensors borrow the buffers for this call. CUDA tensors own device copies.
    pub fn execute(
        self: *CachedMatmul,
        comptime element: ElementType,
        lhs: []const element.Type(),
        rhs: []const element.Type(),
        output: []element.Type(),
    ) !void {
        const state = state_from(self);
        const shape = state.shape;
        if (shape.dtype != element) {
            if (!builtin.is_test) log.err("cached matmul element {s} does not match buffer element {s}", .{ @tagName(shape.dtype), @tagName(element) });
            return error.UnsupportedDType;
        }
        const counts = try element_counts(shape);
        if (lhs.len != counts.lhs or
            rhs.len != counts.rhs or
            output.len != counts.output)
        {
            if (!builtin.is_test) log.err("matmul buffer counts lhs {d}/{d}, rhs {d}/{d}, output {d}/{d} (actual/expected)", .{
                lhs.len, counts.lhs, rhs.len, counts.rhs, output.len, counts.output,
            });
            return error.InvalidShape;
        }

        const allocator = state.allocator;
        var shape_lhs = [_]i64{ shape.m, shape.k };
        var shape_rhs = [_]i64{ shape.k, shape.n };
        var shape_output = [_]i64{ shape.m, shape.n };
        switch (state.target) {
            .cpu => {
                var lhs_tensor = dlpack.Tensor.init_contiguous(element.Type(), @constCast(lhs), &shape_lhs);
                lhs_tensor.dtype = ffi.dlpack_dtype(element.dtype());
                var rhs_tensor = dlpack.Tensor.init_contiguous(element.Type(), @constCast(rhs), &shape_rhs);
                rhs_tensor.dtype = ffi.dlpack_dtype(element.dtype());
                var output_tensor = dlpack.Tensor.init_contiguous(element.Type(), output, &shape_output);
                output_tensor.dtype = ffi.dlpack_dtype(element.dtype());
                var dl_lhs = dlpack.ManagedTensor.borrowing(lhs_tensor);
                var dl_rhs = dlpack.ManagedTensor.borrowing(rhs_tensor);
                var dl_output = dlpack.ManagedTensor.borrowing(output_tensor);
                var tvm_lhs = try tvm_runtime.Tensor.from_dlpack(&dl_lhs);
                defer tvm_lhs.deinit();
                var tvm_rhs = try tvm_runtime.Tensor.from_dlpack(&dl_rhs);
                defer tvm_rhs.deinit();
                var tvm_output = try tvm_runtime.Tensor.from_dlpack(&dl_output);
                defer tvm_output.deinit();
                try state.artifact.invoke(allocator, &.{ tvm_lhs, tvm_rhs, tvm_output });
            },
            .cuda => {
                var tvm_lhs = try tvm_runtime.Tensor.allocate(allocator, std.mem.sliceAsBytes(lhs), &shape_lhs, ffi.dlpack_dtype(element.dtype()), .cuda, state.device.ordinal);
                defer tvm_lhs.deinit();
                var tvm_rhs = try tvm_runtime.Tensor.allocate(allocator, std.mem.sliceAsBytes(rhs), &shape_rhs, ffi.dlpack_dtype(element.dtype()), .cuda, state.device.ordinal);
                defer tvm_rhs.deinit();
                const initial = try allocator.alloc(element.Type(), output.len);
                defer allocator.free(initial);
                @memset(initial, 0);
                var tvm_output = try tvm_runtime.Tensor.allocate(allocator, std.mem.sliceAsBytes(initial), &shape_output, ffi.dlpack_dtype(element.dtype()), .cuda, state.device.ordinal);
                defer tvm_output.deinit();
                try state.artifact.invoke(allocator, &.{ tvm_lhs, tvm_rhs, tvm_output });
                try tvm_output.copy_to_host(allocator, std.mem.sliceAsBytes(output));
            },
        }
    }

    fn state_from(self: *CachedMatmul) *CachedMatmulState {
        return @ptrCast(@alignCast(self));
    }
};

const ElementCounts = struct {
    lhs: usize,
    rhs: usize,
    output: usize,
};

const CacheLocation = struct {
    work: Cache,
    lock: Cache,
    key: artifact.CacheKey,
};

const TuningLock = struct {
    io: std.Io,
    file: std.Io.File,

    fn deinit(self: *TuningLock) void {
        self.file.close(self.io);
        self.* = undefined;
    }
};

fn cache_location(
    io: std.Io,
    artifact_cache: *const Cache,
    shape: Shape,
    compiler_identity: config.BuildIdentity,
    target_description: []const u8,
    create: bool,
) !CacheLocation {
    try validate_shape(shape);
    const base = try artifact_cache.subdir(io, "tvm", .{ .create = create });
    const key = artifact.matmul_cache_key(
        compiler_identity,
        target_description,
        shape.dtype.dtype(),
        shape.m,
        shape.n,
        shape.k,
    );
    var lock_name_buffer: [80]u8 = undefined;
    const lock_name = std.fmt.bufPrint(&lock_name_buffer, "{s}.lock", .{key.slice()}) catch
        unreachable;
    return .{
        .work = try base.subdir(io, key.slice(), .{ .create = create }),
        .lock = try base.join(lock_name),
        .key = key,
    };
}

fn acquire_tuning_lock(io: std.Io, path: *const Cache) !TuningLock {
    const file = std.Io.Dir.cwd().createFile(io, path.path(), .{
        .truncate = false,
        .lock = .exclusive,
        .lock_nonblocking = true,
    }) catch |err| switch (err) {
        error.WouldBlock => return error.TuningInProgress,
        else => return err,
    };
    return .{ .io = io, .file = file };
}

fn reset_tuning_records(io: std.Io, work_cache: *const Cache) !void {
    const cwd = std.Io.Dir.cwd();
    for ([_][]const u8{ "workload.json", "tuning_record.json" }) |name| {
        const path = try work_cache.join(name);
        cwd.deleteFile(io, path.path()) catch |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        };
    }
}

fn validate_shape(shape: Shape) error{InvalidShape}!void {
    if (shape.m <= 0 or shape.n <= 0 or shape.k <= 0)
        return error.InvalidShape;
}

fn element_counts(shape: Shape) error{InvalidShape}!ElementCounts {
    try validate_shape(shape);
    const m_size = std.math.cast(usize, shape.m) orelse return error.InvalidShape;
    const n_size = std.math.cast(usize, shape.n) orelse return error.InvalidShape;
    const k_size = std.math.cast(usize, shape.k) orelse return error.InvalidShape;
    return .{
        .lhs = std.math.mul(usize, m_size, k_size) catch return error.InvalidShape,
        .rhs = std.math.mul(usize, k_size, n_size) catch return error.InvalidShape,
        .output = std.math.mul(usize, m_size, n_size) catch return error.InvalidShape,
    };
}

fn build_ir_module(allocator: std.mem.Allocator, shape: Shape, target: config.TargetKind) !tir.IRModule {
    try validate_shape(shape);
    const lhs_shape = [_]i64{ shape.m, shape.k };
    const rhs_shape = [_]i64{ shape.k, shape.n };

    var lhs = try te.placeholder(allocator, &lhs_shape, shape.dtype.dtype(), "A");
    defer lhs.deinit();
    var rhs = try te.placeholder(allocator, &rhs_shape, shape.dtype.dtype(), "B");
    defer rhs.deinit();
    var output = try te.matmul(allocator, &lhs, &rhs, shape.dtype.matmul_options());
    defer output.deinit();
    var function = try te.create_prim_func(allocator, &.{ &lhs, &rhs, &output });
    defer function.deinit();
    if (target == .cpu) {
        var borrowed = try function.with_buffer_alignment(allocator, @intCast(shape.dtype.dtype().size_in_bytes()));
        defer borrowed.deinit();
        return try tir.IRModule.from_entry(allocator, "main", &borrowed);
    }
    return try tir.IRModule.from_entry(allocator, "main", &function);
}

fn check_cpu_matmul(comptime element: ElementType, shape: Shape, lhs: []const element.Type(), rhs: []const element.Type(), expected: []const element.Type()) !void {
    const allocator = std.testing.allocator;
    try integration_runtime.configure(.{ .surface = .compiler });
    try integration_runtime.ensure_loaded(.compiler);
    var module = try build_ir_module(allocator, shape, .cpu);
    defer module.deinit();
    var target = try tir.Target.create(allocator, .cpu, 0);
    defer target.deinit();
    var compiled = try tvm_compile.lower_and_compile(allocator, &module, target, .cpu);
    defer compiled.deinit();
    var main = try compiled.get_function(allocator, "main", false);
    defer main.deinit();
    const output = try allocator.alloc(element.Type(), expected.len);
    defer allocator.free(output);
    @memset(output, 123);
    var state = CachedMatmulState{
        .allocator = allocator,
        .target = .cpu,
        .device = .{ .platform = .cpu },
        .shape = shape,
        .artifact = .{ .module = compiled, .main_func = main },
    };
    const cached: *CachedMatmul = @ptrCast(&state);
    try cached.execute(element, lhs, rhs, output);
    try std.testing.expectEqualSlices(element.Type(), expected, output);
}

test "CachedMatmul execute borrows CPU buffers against direct reference" {
    const lhs = [_]f32{ 1, -2, 3, 4, -5, 6, 7, -8 };
    const rhs = [_]f32{ 1, 2, -3, 4, -5, 6, -7, 8, 9, 10, 11, -12 };
    var reference: [6]f32 = @splat(0);
    for (0..2) |i| for (0..3) |j| {
        for (0..4) |k| reference[i * 3 + j] += lhs[i * 4 + k] * rhs[k * 3 + j];
    };
    try check_cpu_matmul(.f32, .{ .m = 2, .n = 3, .k = 4 }, &lhs, &rhs, &reference);
}

test "CachedMatmul execute borrows naturally aligned offset CPU buffers" {
    const allocator = std.testing.allocator;
    try integration_runtime.configure(.{ .surface = .compiler });
    try integration_runtime.ensure_loaded(.compiler);
    const shape: Shape = .{ .m = 2, .n = 3, .k = 4 };
    var module = try build_ir_module(allocator, shape, .cpu);
    defer module.deinit();
    var target = try tir.Target.create(allocator, .cpu, 0);
    defer target.deinit();
    var compiled = try tvm_compile.lower_and_compile(allocator, &module, target, .cpu);
    defer compiled.deinit();
    var main = try compiled.get_function(allocator, "main", false);
    defer main.deinit();
    var state = CachedMatmulState{
        .allocator = allocator,
        .target = .cpu,
        .device = .{ .platform = .cpu },
        .shape = shape,
        .artifact = .{ .module = compiled, .main_func = main },
    };
    const cached: *CachedMatmul = @ptrCast(&state);
    var lhs_storage: [10]f32 align(64) = .{ 123, 1, -2, 3, 4, -5, 6, 7, -8, 123 };
    var rhs_storage: [14]f32 align(64) = .{ 123, 1, 2, -3, 4, -5, 6, -7, 8, 9, 10, 11, -12, 123 };
    var output_storage: [8]f32 align(64) = @splat(123);
    const lhs = lhs_storage[1..9];
    const rhs = rhs_storage[1..13];
    const output = output_storage[1..7];
    for ([_][]f32{ lhs, rhs, output }) |buffer| {
        try std.testing.expectEqual(@as(usize, 4), @intFromPtr(buffer.ptr) % 64);
        try std.testing.expectEqual(@as(usize, 0), @intFromPtr(buffer.ptr) % @alignOf(f32));
    }
    var expected: [6]f32 = @splat(0);
    for (0..2) |i| for (0..3) |j| {
        for (0..4) |k| expected[i * 3 + j] += lhs[i * 4 + k] * rhs[k * 3 + j];
    };
    try cached.execute(.f32, lhs, rhs, output);
    try std.testing.expectEqualSlices(f32, &expected, output);
    try std.testing.expectEqual(@as(f32, 123), output_storage[0]);
    try std.testing.expectEqual(@as(f32, 123), output_storage[7]);
    try std.testing.expectEqualSlices(f32, &.{ 1, -2, 3, 4, -5, 6, 7, -8 }, lhs);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, -3, 4, -5, 6, -7, 8, 9, 10, 11, -12 }, rhs);
}

test "TVM f16 matmul multiplies and reduces in f32 before output rounding" {
    try check_cpu_matmul(.f16, .{ .m = 1, .n = 1, .k = 2, .dtype = .f16 }, &.{ 256, -256 }, &.{ 256, 256 }, &.{0});
    const ones: [4096]f16 = @splat(1);
    try check_cpu_matmul(.f16, .{ .m = 1, .n = 1, .k = 4096, .dtype = .f16 }, &ones, &ones, &.{4096});
    try check_cpu_matmul(.f16, .{ .m = 1, .n = 1, .k = 2, .dtype = .f16 }, &.{ 1, 0x1p-11 }, &.{ 1, 1 }, &.{1});
    try check_cpu_matmul(.f16, .{ .m = 1, .n = 1, .k = 3, .dtype = .f16 }, &.{ 1, 0x1p-11, 0x1p-12 }, &.{ 1, 1, 1 }, &.{1 + 0x1p-10});
    try check_cpu_matmul(.f32, .{ .m = 1, .n = 1, .k = 3 }, &.{ 1, 0x1p-11, 0x1p-12 }, &.{ 1, 1, 1 }, &.{1 + 0x1p-11 + 0x1p-12});
}

test "TVM generates an sm80 f16 matmul design space" {
    const cuda_intrinsics = @import("cuda_intrinsics.zig");
    const logger_callback = struct {
        fn call(
            _: ?*anyopaque,
            _: *const ffi.CallbackArgs,
        ) !ffi.CallbackOutput {
            return .none;
        }
    }.call;

    const intrinsic_path = std.process.Environ.getAlloc(
        std.testing.environ,
        std.testing.allocator,
        config.cuda_intrinsics_path_env,
    ) catch |err| switch (err) {
        error.EnvironmentVariableMissing => return error.SkipZigTest,
        else => return err,
    };
    defer std.testing.allocator.free(intrinsic_path);
    try integration_runtime.configure(.{ .surface = .compiler });
    try integration_runtime.ensure_loaded(.compiler);
    try cuda_intrinsics.ensure_registered(
        std.testing.io,
        std.testing.allocator,
        intrinsic_path,
    );

    var ir_module = try build_ir_module(
        std.testing.allocator,
        .{ .m = 128, .n = 128, .k = 128, .dtype = .f16 },
        .cuda,
    );
    defer ir_module.deinit();
    var target = try tir.Target.from_description(
        std.testing.allocator,
        "cuda -arch=sm_80 -max_shared_memory_per_block=49152 " ++
            "-max_threads_per_block=1024 -thread_warp_size=32 " ++
            "-registers_per_block=65536 -l2_cache_size_bytes=41943040",
    );
    defer target.deinit();
    var space_generator = try ms.SpaceGenerator.post_order_apply(std.testing.allocator);
    defer space_generator.deinit();
    var search_strategy = try ms.SearchStrategy.replay_trace(std.testing.allocator, .{});
    defer search_strategy.deinit();
    var logger = try ffi.PackedFunction.create(null, logger_callback, null);
    defer logger.deinit();
    var context = try ms.TuneContext.init(std.testing.allocator, .{
        .module = ir_module,
        .target = target,
        .generator = &space_generator,
        .strategy = &search_strategy,
        .task_name = "main",
        .logger = &logger,
    });
    defer context.deinit();

    var schedules = try space_generator.generate(
        std.testing.allocator,
        ir_module,
    );
    defer schedules.deinit();
    try std.testing.expect(try schedules.len(std.testing.allocator) > 0);
    try std.testing.expect(try design_space_uses_tensorization(
        std.testing.allocator,
        schedules,
    ));
}

fn design_space_uses_tensorization(
    allocator: std.mem.Allocator,
    schedules: ms.Schedules,
) !bool {
    for (0..try schedules.len(allocator)) |schedule_index| {
        var schedule = try schedules.get(allocator, schedule_index);
        defer schedule.deinit();
        var trace = try schedule.python_trace(allocator);
        defer trace.deinit();
        for (trace.lines) |line| {
            if (std.mem.indexOf(u8, line, "meta_schedule.auto_tensorize") != null)
                return true;
        }
    }
    return false;
}

test element_counts {
    try std.testing.expectEqualDeep(
        ElementCounts{ .lhs = 6, .rhs = 6, .output = 4 },
        try element_counts(.{ .m = 2, .n = 2, .k = 3 }),
    );
    try std.testing.expectError(
        error.InvalidShape,
        element_counts(.{ .m = 0, .n = 2, .k = 3 }),
    );
}

fn check_f32_arithmetic(module: *const tir.IRModule, allocator: std.mem.Allocator) !void {
    const json = try module.dupe_json(allocator);
    defer allocator.free(json);
    const Data = struct {
        dtype: ?[]const u8 = null,
        op: ?usize = null,
        string: ?[]const u8 = null,
        pub fn jsonParse(gpa: std.mem.Allocator, source: anytype, options: std.json.ParseOptions) !@This() {
            if (try source.peekNextTokenType() == .object_begin) {
                const attrs = try std.json.innerParse(struct { dtype: ?[]const u8 = null, op: ?usize = null }, gpa, source, options);
                return .{ .dtype = attrs.dtype, .op = attrs.op };
            }
            if (try source.peekNextTokenType() == .string) return .{ .string = try std.json.innerParse([]const u8, gpa, source, options) };
            try source.skipValue();
            return .{};
        }
    };
    const Graph = struct { nodes: []struct { type: []const u8, data: Data = .{} } };
    const parsed = try std.json.parseFromSlice(Graph, allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    var products: usize = 0;
    var half_conversions: usize = 0;
    var widened_inputs: usize = 0;
    for (parsed.value.nodes) |node| {
        const dtype = node.data.dtype orelse continue;

        if (std.mem.eql(u8, node.type, "tir.Mul") and std.mem.startsWith(u8, dtype, "float")) {
            try std.testing.expectEqualStrings("float32", dtype);
            products += 1;
        }
        if (std.mem.eql(u8, node.type, "tir.Call") and std.mem.startsWith(u8, dtype, "float")) {
            try std.testing.expectEqualStrings("float32", dtype);
            const op = node.data.op orelse return error.MissingIntrinsic;
            const intrinsic = parsed.value.nodes[op].data.string orelse return error.MissingIntrinsic;
            try std.testing.expectEqualStrings("tir.call_llvm_pure_intrin", intrinsic);
            products += 1;
        }
        if (std.mem.eql(u8, node.type, "tir.Cast") and std.mem.eql(u8, dtype, "float32")) widened_inputs += 1;
        if (std.mem.eql(u8, node.type, "tir.Add") and std.mem.startsWith(u8, dtype, "float")) {
            try std.testing.expectEqualStrings("float32", dtype);
        }
        if (std.mem.eql(u8, node.type, "tir.Cast") and std.mem.eql(u8, dtype, "float16")) half_conversions += 1;
    }
    try std.testing.expect(products > 0);
    try std.testing.expectEqual(@as(usize, 2), widened_inputs);
    try std.testing.expectEqual(@as(usize, 1), half_conversions);
}

test "TVM f16 TIR carries f32 products reduction and one output cast" {
    const allocator = std.testing.allocator;
    try integration_runtime.configure(.{ .surface = .compiler });
    try integration_runtime.ensure_loaded(.compiler);
    var module = try build_ir_module(allocator, .{ .m = 2, .n = 3, .k = 4, .dtype = .f16 }, .cpu);
    defer module.deinit();
    try check_f32_arithmetic(&module, allocator);
    var target = try tir.Target.create(allocator, .cpu, 0);
    defer target.deinit();
    var compiled = try tvm_compile.lower_and_compile(allocator, &module, target, .cpu);
    defer compiled.deinit();
    try check_f32_arithmetic(&module, allocator);
}

test "TVM publication measures the selected file and preserves current on failure" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    try integration_runtime.configure(.{ .surface = .compiler });
    try integration_runtime.ensure_loaded(.compiler);
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    const root = try temporary.dir.realPathFileAlloc(io, ".", allocator);
    defer allocator.free(root);
    var environ = try std.process.Environ.createMap(std.testing.environ, allocator);
    defer environ.deinit();
    const compile_config = try config.CompileConfig.from_environ(&environ, .cpu);
    const cache = try Cache.init(io, &environ, .{ .root = root });
    const shape: Shape = .{ .m = 2, .n = 3, .k = 4 };
    var resolved = try tir.Target.resolve(allocator, compile_config.target, .{ .platform = .cpu });
    defer resolved.deinit();
    const location = try cache_location(io, &cache, shape, compile_config.build_identity(), resolved.description(), true);
    const protocol: tvm_runtime.TimeEvaluatorOptions = .{ .number = 1, .repeats = 2, .min_repeat_ms = 0 };
    const partitions = try location.work.subdir(io, "measurements", .{});
    const key = artifact.protocol_key(protocol);
    const partition = try partitions.subdir(io, key.slice(), .{});
    var workload_file = try partition.join("workload.json");
    var record_file = try partition.join("tuning_record.json");
    var database = try ms.Database.open_json(allocator, workload_file.pathZ(), record_file.pathZ(), .{});
    defer database.deinit();
    var module = try build_ir_module(allocator, shape, .cpu);
    defer module.deinit();
    var workload = try database.commit_workload(allocator, module);
    defer workload.deinit();
    var schedule = try ms.Schedule.create(allocator, module, .{ .seed = 1 });
    defer schedule.deinit();
    var block = try schedule.get_block(allocator, "accumulator", "main");
    defer block.deinit();
    var record = try ms.TuningRecord.from_schedule(allocator, &schedule, &workload, resolved.target(), &.{1e-9});
    defer record.deinit();
    try database.commit_record(allocator, &record);
    var top = try database.top_k(allocator, &workload, 1);
    defer top.deinit();
    try std.testing.expectEqual(@as(usize, 1), try top.len(allocator));
    {
        var reused = try ms.Database.open_json(allocator, workload_file.pathZ(), record_file.pathZ(), .{});
        defer reused.deinit();
        var reused_workload = try reused.commit_workload(allocator, module);
        defer reused_workload.deinit();
        var reused_top = try reused.top_k(allocator, &reused_workload, 1);
        defer reused_top.deinit();
        try std.testing.expectEqual(@as(usize, 1), try reused_top.len(allocator));
    }
    inline for (std.meta.fields(tvm_runtime.TimeEvaluatorOptions)) |field| {
        var changed_protocol = protocol;
        @field(changed_protocol, field.name) += 1;
        const changed_key = artifact.protocol_key(changed_protocol);
        const changed_partition = try partitions.subdir(io, changed_key.slice(), .{});
        var changed_workload_file = try changed_partition.join("workload.json");
        var changed_record_file = try changed_partition.join("tuning_record.json");
        var changed_database = try ms.Database.open_json(allocator, changed_workload_file.pathZ(), changed_record_file.pathZ(), .{});
        defer changed_database.deinit();
        var changed_workload = try changed_database.commit_workload(allocator, module);
        defer changed_workload.deinit();
        var changed_top = try changed_database.top_k(allocator, &changed_workload, 1);
        defer changed_top.deinit();
        try std.testing.expectEqual(@as(usize, 0), try changed_top.len(allocator));
    }
    {
        const other = try cache_location(io, &cache, shape, .from_external("other-external-inputs"), resolved.description(), true);
        const other_measurements = try other.work.subdir(io, "measurements", .{});
        const other_partition = try other_measurements.subdir(io, key.slice(), .{});
        var other_workload_file = try other_partition.join("workload.json");
        var other_record_file = try other_partition.join("tuning_record.json");
        var other_database = try ms.Database.open_json(allocator, other_workload_file.pathZ(), other_record_file.pathZ(), .{});
        defer other_database.deinit();
        var other_workload = try other_database.commit_workload(allocator, module);
        defer other_workload.deinit();
        var other_top = try other_database.top_k(allocator, &other_workload, 1);
        defer other_top.deinit();
        try std.testing.expectEqual(@as(usize, 0), try other_top.len(allocator));
    }
    var selected = try top.get(allocator, 0);
    defer selected.deinit();
    var replayed = try selected.replay_module(allocator);
    defer replayed.deinit();
    var compiled = try tvm_compile.lower_and_compile(allocator, &replayed, resolved.target(), .cpu);
    defer compiled.deinit();
    const result = try publish_module(io, allocator, &location.work, shape, compile_config, resolved.description(), .{ .platform = .cpu }, protocol, compiled);
    try std.testing.expectEqual(@as(usize, 2), result.sample_count);
    const bytes = (try artifact.read_cached_bytes(io, allocator, &location.work)).?;
    defer allocator.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    try std.testing.expectEqualStrings(&std.fmt.bytesToHex(digest, .lower), &result.artifact_digest);
    const generations = try location.work.join("generations");
    const generation = try generations.join(&result.generation);
    const evidence_file = try generation.join("evidence.json");
    const evidence = try std.Io.Dir.cwd().readFileAlloc(io, evidence_file.path(), allocator, .limited(65536));
    defer allocator.free(evidence);
    const Evidence = struct {
        generation: []const u8,
        artifact: []const u8,
        artifact_digest: []const u8,
        protocol: tvm_runtime.TimeEvaluatorOptions,
        samples_seconds: []f64,
        lowering_identity: []const u8,
        external_identity: []const u8,
        build_identity: []const u8,
    };
    const parsed = try std.json.parseFromSlice(Evidence, allocator, evidence, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings(result.artifact_name, parsed.value.artifact);
    try std.testing.expectEqualStrings(&result.generation, parsed.value.generation);
    try std.testing.expectEqualStrings(&result.artifact_digest, parsed.value.artifact_digest);
    try std.testing.expectEqualDeep(protocol, parsed.value.protocol);
    var sum: f64 = 0;
    for (parsed.value.samples_seconds) |sample| sum += sample;
    try std.testing.expectEqual(sum / @as(f64, @floatFromInt(result.sample_count)) * 1e6, result.best_time_us);
    const failed_generation = try artifact.Generation.create(io, &location.work);
    try failed_generation.stage(compiled, io, allocator, .cpu, compile_config.linker);
    defer std.Io.Dir.cwd().deleteTree(io, failed_generation.cache.path()) catch {};
    try std.testing.expectError(error.InvalidTimingResult, finish_publication(io, allocator, &location.work, &failed_generation, shape, compile_config, resolved.description(), protocol, &.{ std.math.nan(f64), 1.0 }));
    const pointer = try location.work.join("current");
    const current = try std.Io.Dir.cwd().readFileAlloc(io, pointer.path(), allocator, .limited(33));
    defer allocator.free(current);
    try std.testing.expectEqualStrings(&result.generation, current);
    if (std.process.Environ.getAlloc(std.testing.environ, allocator, "ZG_SCRATCH_ROOT")) |scratch| {
        defer allocator.free(scratch);
        try std.Io.Dir.cwd().createDirPath(io, scratch);
        const destination = try std.fs.path.join(allocator, &.{ scratch, "tvm-publication-evidence.json" });
        defer allocator.free(destination);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = destination, .data = evidence });
    } else |err| switch (err) {
        error.EnvironmentVariableMissing => {},
        else => return err,
    }
}

test "CachedMatmul execute rejects mismatched elements and buffers before invocation" {
    var state = CachedMatmulState{
        .allocator = std.testing.allocator,
        .target = .cpu,
        .device = .{ .platform = .cpu },
        .shape = .{ .m = 1, .n = 1, .k = 1, .dtype = .f16 },
        .artifact = undefined,
    };
    const cached: *CachedMatmul = @ptrCast(&state);
    var output: [1]f32 = undefined;
    try std.testing.expectError(error.UnsupportedDType, cached.execute(.f32, &.{1}, &.{1}, &output));
    var short: [0]f16 = .{};
    try std.testing.expectError(error.InvalidShape, cached.execute(.f16, &.{1}, &.{1}, &short));
}
