//! MetaSchedule autotuning for TVM.
//!
//! Runs TVM's MetaSchedule search for a given IRModule and target.
//! Builder and runner callbacks receive provider state through TVM's
//!  userdata pointer. Schedule timings rank implementations within TVM. The
//!  Zigrad tuning resolver compares the selected provider implementation with
//!  the unreplaced callable.
const std = @import("std");
const device = @import("device");
const tir = @import("../c/tvm/tir.zig");
const runtime = @import("../c/tvm/runtime.zig");
const ms = @import("../c/tvm/meta_schedule.zig");
const compile = @import("../c/tvm/compile.zig");
const ffi = @import("../c/tvm/ffi.zig");
const container = @import("../c/tvm/container.zig");
const dlpack = @import("../c/dlpack.zig");
const DType = @import("dtype").DType;
const Cache = @import("../cache.zig").Cache;
const build_options = @import("build_options");
const config = @import("config.zig");
const cuda_intrinsics = @import("cuda_intrinsics.zig");
const integration_runtime = @import("runtime.zig");
const export_mod = @import("export.zig");
const Linker = @import("../toolchain/linker.zig").Linker;
const Value = ffi.Value;
const OwnedValue = ffi.OwnedValue;
const Array = container.Array;
const IRModule = tir.IRModule;
const RuntimeModule = runtime.RuntimeModule;
const Target = tir.Target;
const Tensor = runtime.Tensor;
const TargetKind = @import("config.zig").TargetKind;
const nvrtc_callback = if (build_options.has_nvrtc) @import("nvrtc_callback.zig") else struct {};

const log = std.log.scoped(.@"zg/tvm_tune");

pub const TuneOpts = struct {
    /// Target-specific compiler inputs resolved by application composition.
    compile: config.CompileConfig,

    /// Device used for candidate compilation and measurement.
    device: device.Device,

    /// NVRTC architecture resolved from the TVM CUDA target.
    gpu_arch: ?[]const u8,

    /// Directory containing this workload's tuning state and candidates.
    work_cache: *const Cache,

    /// Maximum measured candidates.
    max_trials: u32 = 64,

    /// Candidates submitted per tuning iteration.
    trials_per_iter: u32 = 16,

    /// TVM timing policy applied to each compiled schedule.
    measurement: runtime.TimeEvaluatorOptions = .{},
};

/// Highest-ranked schedule and its measurements.
pub const Result = struct {
    /// Scheduled module selected by TVM's database ranking.
    module: IRModule,
    /// Mean of the selected record's measurements.
    mean_time_seconds: f64,
    /// Number of measurements contributing to the mean.
    sample_count: usize,

    /// Release the selected scheduled module.
    pub fn deinit(self: *Result) void {
        self.module.deinit();
        self.* = undefined;
    }
};

/// State passed to MetaSchedule builder and runner callbacks.
const TuneState = struct {
    io: std.Io,
    allocator: std.mem.Allocator,
    target: Target,
    target_kind: TargetKind,
    linker: Linker,
    device_ordinal: i32,
    candidate_cache: Cache,
    build_counter: u32 = 0,
    candidate_batch_max: u32,
    /// Tensor shapes for the workload (A, B, C for matmul).
    tensor_shapes: []const []const i64,
    tensor_dtype: DType,
    measurement: runtime.TimeEvaluatorOptions,
};

/// Run MetaSchedule and return its highest-ranked schedule.
///
/// ReplayTrace samples schedules from TVM's target-selected design space. The
///  builder compiles candidates to shared libraries and the runner measures
///  them. TVM persists workloads and measurements in `opts.work_cache`. The
///  orchestration corresponds to `python/tvm/meta_schedule/tune.py::tune_tasks`.
pub fn tune(
    /// I/O context used by filesystem and timing operations.
    io: std.Io,
    /// Allocator used for TVM API and callback state.
    allocator: std.mem.Allocator,
    /// TIR module supplied to MetaSchedule.
    ir_mod: IRModule,
    /// TVM compilation target.
    target: Target,
    /// Element type shared by the matrix-multiply inputs and output.
    tensor_dtype: DType,
    /// Input and output tensor shapes used by the runner.
    tensor_shapes: []const []const i64,
    /// Compiler, device, cache, and search limits.
    opts: TuneOpts,
) !Result {
    try validate_tuning_limits(
        opts.max_trials,
        opts.trials_per_iter,
        opts.measurement,
    );

    try integration_runtime.ensure_loaded(.compiler);
    const kind = opts.compile.target;

    if (kind == .cuda) {
        if (comptime build_options.has_nvrtc) {
            const nvrtc_config = opts.compile.nvrtc orelse {
                log.err("TVM CUDA tuning requires resolved NVRTC configuration", .{});
                return error.MissingNvrtcConfig;
            };
            const gpu_arch = opts.gpu_arch orelse {
                log.err("TVM CUDA target has no resolved NVRTC architecture", .{});
                return error.MissingNvrtcArchitecture;
            };
            nvrtc_callback.register(
                nvrtc_config,
                gpu_arch,
            ) catch |err| {
                log.err("failed to register NVRTC callback: {s}", .{@errorName(err)});
                return err;
            };
        } else {
            log.err("TVM CUDA tuning requires the opt-in NVRTC integration", .{});
            return error.NvrtcDisabled;
        }
        try cuda_intrinsics.ensure_registered(
            io,
            allocator,
            opts.compile.cuda_intrinsics_path orelse
                return error.MissingCudaIntrinsics,
        );
    }

    const work_dir = opts.work_cache.path();
    try std.Io.Dir.cwd().createDirPath(io, work_dir);

    // Candidate libraries serve the runner only while this invocation is live.
    //  Schedule traces and measurements remain in TVM's database.
    const candidate_cache = try opts.work_cache.join("candidates");
    try std.Io.Dir.cwd().deleteTree(io, candidate_cache.path());
    try std.Io.Dir.cwd().createDirPath(io, candidate_cache.path());
    defer std.Io.Dir.cwd().deleteTree(io, candidate_cache.path()) catch |err| {
        log.warn("failed to remove candidate directory: {s}", .{@errorName(err)});
    };

    try register_cpu_count();

    var space_gen = try ms.SpaceGenerator.post_order_apply(allocator);
    defer space_gen.deinit();
    log.debug("created SpaceGenerator", .{});

    var search_strategy = try ms.SearchStrategy.replay_trace(allocator, .{});
    defer search_strategy.deinit();
    log.debug("created SearchStrategy", .{});

    var workload = try opts.work_cache.join("workload.json");
    const workload_path = workload.pathZ();
    var record = try opts.work_cache.join("tuning_record.json");
    const record_path = record.pathZ();

    var database = try ms.Database.open_json(
        allocator,
        workload_path,
        record_path,
        .{},
    );
    defer database.deinit();
    log.debug("created JSONDatabase", .{});

    var logger_value = try make_noop_callback();
    defer logger_value.deinit();

    var tune_context = try ms.TuneContext.init(allocator, .{
        .module = ir_mod,
        .target = target,
        .generator = &space_gen,
        .strategy = &search_strategy,
        .task_name = "main",
        .logger = &logger_value,
    });
    defer tune_context.deinit();
    log.debug("created TuneContext", .{});

    var state = TuneState{
        .io = io,
        .allocator = allocator,
        .target = target,
        .target_kind = kind,
        .linker = opts.compile.linker,
        .device_ordinal = opts.device.ordinal,
        .candidate_cache = candidate_cache,
        .candidate_batch_max = opts.trials_per_iter,
        .tensor_shapes = tensor_shapes,
        .tensor_dtype = tensor_dtype,
        .measurement = opts.measurement,
    };

    var builder_func = try ffi.PackedFunction.create(@ptrCast(&state), build_callback, null);
    defer builder_func.deinit();
    var builder = try ms.Builder.from_callback(allocator, &builder_func);
    defer builder.deinit();
    log.debug("created callback builder", .{});

    var runner_func = try ffi.PackedFunction.create(@ptrCast(&state), run_callback, null);
    defer runner_func.deinit();
    var runner = try ms.Runner.from_callback(allocator, &runner_func);
    defer runner.deinit();
    log.debug("created callback runner", .{});

    var task_scheduler = try ms.TaskScheduler.gradient_based(allocator, .{
        .logger = &logger_value,
    });
    defer task_scheduler.deinit();
    log.debug("created TaskScheduler", .{});

    log.info("starting tuning ({d} max trials, {d} per iter)...", .{ opts.max_trials, opts.trials_per_iter });

    var add_to_db = try ms.MeasureCallback.add_to_database(allocator);
    defer add_to_db.deinit();

    const max_trials: i64 = @intCast(opts.max_trials);
    task_scheduler.tune(allocator, .{
        .contexts = &.{&tune_context},
        .weights = &.{1.0},
        .max_trials_global = max_trials,
        .max_trials_per_task = max_trials,
        .trials_per_iter = @intCast(opts.trials_per_iter),
        .builder = &builder,
        .runner = &runner,
        .callbacks = &.{&add_to_db},
        .database = &database,
    }) catch |err| {
        log.err("TaskSchedulerTune failed: {s}", .{@errorName(err)});
        return err;
    };

    var committed_workload = try database.commit_workload(allocator, ir_mod);
    defer committed_workload.deinit();
    var top = try database.top_k(allocator, &committed_workload, 1);
    defer top.deinit();
    if (try top.len(allocator) != 1) return error.NoTuningRecords;

    var best_record = try top.get(allocator, 0);
    defer best_record.deinit();
    const samples = try best_record.dupe_run_seconds(allocator);
    defer allocator.free(samples);
    var sum: f64 = 0.0;
    for (samples) |sample| sum += sample;
    const mean = sum / @as(f64, @floatFromInt(samples.len));

    const scheduled_module = try database.query_module(
        allocator,
        ir_mod,
        target,
        "main",
    ) orelse return error.NoTuningRecords;
    log.info("tuning complete: {d} samples, {d:.2} us mean", .{
        samples.len,
        mean * 1e6,
    });
    return .{
        .module = scheduled_module,
        .mean_time_seconds = mean,
        .sample_count = samples.len,
    };
}

/// Builder callback: compiles TIR candidates to .so artifacts.
fn build_callback(
    self_ptr: ?*anyopaque,
    args: *const ffi.CallbackArgs,
) !ffi.CallbackOutput {
    const state: *TuneState = @ptrCast(@alignCast(self_ptr orelse {
        log.err("build_callback: null state", .{});
        return error.MissingCallbackState;
    }));

    if (args.len() != 1) {
        log.err("build_callback: expected 1 arg, got {d}", .{args.len()});
        return error.InvalidCallbackArity;
    }

    const result = build_callback_impl(state, args.get(0)) catch |err| {
        log.err("build_callback failed: {s}", .{@errorName(err)});
        return err;
    };
    return .{ .owned = result };
}

fn build_callback_impl(
    state: *TuneState,
    inputs_value: Value,
) !OwnedValue {
    const allocator = state.allocator;

    var inputs = try Array.retain(inputs_value);
    defer inputs.deinit();

    const num_inputs = try inputs.len(allocator);
    if (num_inputs > state.candidate_batch_max) return error.CandidateBatchTooLarge;
    log.info("building {d} candidates", .{num_inputs});

    var results_list = std.ArrayList(OwnedValue).empty;
    defer {
        for (results_list.items) |*value| value.deinit();
        results_list.deinit(allocator);
    }

    for (0..num_inputs) |i| {
        // Build ids name temporary candidate libraries.
        const build_id = @atomicRmw(u32, &state.build_counter, .Add, 1, .seq_cst);
        var input = try inputs.get(allocator, i);
        defer input.deinit();
        const candidate = build_candidate(state, input.borrow(), build_id) catch |err| blk: {
            log.warn("candidate {d} build failed: {s}", .{ build_id, @errorName(err) });
            break :blk try make_builder_error(allocator, @errorName(err));
        };
        try results_list.append(allocator, candidate);
        log.debug("built trial {d}/{d} (candidate {d})", .{ i + 1, num_inputs, build_id });
    }

    var results_arr = try array_from_owned(allocator, results_list.items);
    return results_arr.take_value();
}

/// Compile one `BuilderInput` from `python/tvm/meta_schedule/builder/builder.py`.
fn build_candidate(state: *TuneState, input_value: Value, build_id: u32) !OwnedValue {
    const allocator = state.allocator;
    const input = try ms.BuilderInput.from_value(input_value);
    var ir_mod = try input.module();
    defer ir_mod.deinit();

    var compiled_module = try compile.lower_and_compile(
        allocator,
        &ir_mod,
        state.target,
        state.target_kind,
    );
    defer compiled_module.deinit();

    var name_buffer: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buffer, "candidate_{d}.so", .{build_id}) catch
        unreachable;
    var artifact_path = try state.candidate_cache.join(name);
    const artifact_path_z = artifact_path.pathZ();
    try export_mod.export_shared(
        compiled_module,
        state.io,
        allocator,
        artifact_path_z,
        state.target_kind,
        state.linker,
    );
    var build_result = try ms.BuilderResult.init(allocator, .{ .success = artifact_path_z });
    return build_result.take_value();
}

/// Runner callback: loads and benchmarks compiled .so artifacts.
fn run_callback(
    self_ptr: ?*anyopaque,
    args: *const ffi.CallbackArgs,
) !ffi.CallbackOutput {
    const state: *TuneState = @ptrCast(@alignCast(self_ptr orelse {
        log.err("run_callback: null state", .{});
        return error.MissingCallbackState;
    }));

    if (args.len() != 1) {
        log.err("run_callback: expected 1 arg, got {d}", .{args.len()});
        return error.InvalidCallbackArity;
    }

    const result = run_callback_impl(state, args.get(0)) catch |err| {
        log.err("run_callback failed: {s}", .{@errorName(err)});
        return err;
    };
    return .{ .owned = result };
}

fn run_callback_impl(
    state: *TuneState,
    inputs_value: Value,
) !OwnedValue {
    const allocator = state.allocator;

    var inputs = try Array.retain(inputs_value);
    defer inputs.deinit();

    const num_inputs = try inputs.len(allocator);
    if (num_inputs > state.candidate_batch_max) return error.CandidateBatchTooLarge;
    log.info("running {d} candidates", .{num_inputs});

    var results_list = std.ArrayList(OwnedValue).empty;
    defer {
        for (results_list.items) |*value| value.deinit();
        results_list.deinit(allocator);
    }

    for (0..num_inputs) |i| {
        var input = try inputs.get(allocator, i);
        defer input.deinit();
        const future = run_candidate(state, input.borrow()) catch |err| blk: {
            log.warn("candidate {d} measurement failed: {s}", .{ i, @errorName(err) });
            break :blk try make_runner_error(allocator, @errorName(err));
        };
        try results_list.append(allocator, future);
    }

    var results_arr = try array_from_owned(allocator, results_list.items);
    return results_arr.take_value();
}

/// Measure one `RunnerInput` from `python/tvm/meta_schedule/runner/runner.py`.
fn run_candidate(state: *TuneState, input_value: Value) !OwnedValue {
    const allocator = state.allocator;
    const input = try ms.RunnerInput.from_value(input_value);
    const path = try input.dupe_artifact_path(allocator);
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var loaded = try RuntimeModule.load_from_file(allocator, path_z);
    defer loaded.deinit();
    const run_times = try benchmark_kernel(state, loaded);
    defer allocator.free(run_times);
    return try make_runner_success(allocator, run_times);
}

/// Measure a compiled kernel with TVM's target timer.
///
/// The returned repeat averages belong to `state.allocator`.
fn benchmark_kernel(state: *TuneState, module: RuntimeModule) ![]f64 {
    const allocator = state.allocator;
    const dev_type: dlpack.DeviceType = switch (state.target_kind) {
        .cpu => .cpu,
        .cuda => .cuda,
    };

    var tensors = std.ArrayList(Tensor).empty;
    defer {
        for (tensors.items) |*t| t.deinit();
        tensors.deinit(allocator);
    }

    for (state.tensor_shapes) |shape| {
        const dtype = try dlpack_dtype(state.tensor_dtype);
        const bytes = dlpack.byte_count(shape, dtype) catch
            return error.InvalidTensorLayout;

        const tensor = blk: {
            const data = try allocator.alloc(u8, bytes);
            defer allocator.free(data);
            @memset(data, 0);

            break :blk try Tensor.allocate(
                allocator,
                data,
                shape,
                dtype,
                dev_type,
                state.device_ordinal,
            );
        };
        try tensors.append(allocator, tensor);
    }

    var evaluator = try module.time_evaluator(
        allocator,
        "main",
        dev_type,
        state.device_ordinal,
        state.measurement,
    );
    defer evaluator.deinit();
    return try evaluator.measure(allocator, tensors.items);
}

fn dlpack_dtype(dtype: DType) !dlpack.DataType {
    return switch (dtype) {
        .f16 => .f16_,
        .f32 => .f32_,
        else => error.UnsupportedDType,
    };
}

fn validate_tuning_limits(
    max_trials: u32,
    trials_per_iter: u32,
    measurement: runtime.TimeEvaluatorOptions,
) !void {
    if (max_trials == 0) return error.InvalidTrialCount;
    if (trials_per_iter == 0) return error.InvalidTrialCount;
    try measurement.validate();
}

test validate_tuning_limits {
    try std.testing.expectError(
        error.InvalidTrialCount,
        validate_tuning_limits(0, 1, .{}),
    );
    try std.testing.expectError(
        error.InvalidTrialCount,
        validate_tuning_limits(1, 0, .{}),
    );
    try std.testing.expectError(
        error.InvalidTimeEvaluatorOptions,
        validate_tuning_limits(1, 1, .{ .repeats = 0 }),
    );
    try validate_tuning_limits(1, 2, .{});
}

fn make_noop_callback() !ffi.PackedFunction {
    const noop = struct {
        fn f(_: ?*anyopaque, _: *const ffi.CallbackArgs) !ffi.CallbackOutput {
            return .none;
        }
    }.f;
    return try ffi.PackedFunction.create(null, noop, null);
}

fn register_cpu_count() !void {
    const cpu_count_cb = struct {
        fn f(_: ?*anyopaque, args: *const ffi.CallbackArgs) !ffi.CallbackOutput {
            if (args.len() != 0) return error.InvalidCallbackArity;
            return .{ .integer = @intCast(std.Thread.getCpuCount() catch 1) };
        }
    }.f;

    var function = try ffi.PackedFunction.create(null, cpu_count_cb, null);
    defer function.deinit();
    try ms.register_cpu_count_callback(&function);
}

fn make_builder_error(allocator: std.mem.Allocator, msg: []const u8) !OwnedValue {
    const msg_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{msg}, 0);
    defer allocator.free(msg_z);
    var result = try ms.BuilderResult.init(allocator, .{ .failure = msg_z });
    return result.take_value();
}

/// Create a RunnerFuture wrapping a RunnerResult with an error message.
fn make_runner_error(allocator: std.mem.Allocator, msg: []const u8) !OwnedValue {
    const msg_z = try std.fmt.allocPrintSentinel(allocator, "{s}", .{msg}, 0);
    defer allocator.free(msg_z);
    var runner_result = try ms.RunnerResult.init(allocator, .{ .failure = msg_z });
    defer runner_result.deinit();
    var future = try ms.RunnerFuture.completed(allocator, &runner_result);
    return future.take_value();
}

/// Create a RunnerFuture wrapping a RunnerResult with timing data.
fn make_runner_success(allocator: std.mem.Allocator, run_secs: []const f64) !OwnedValue {
    var runner_result = try ms.RunnerResult.init(allocator, .{ .success = run_secs });
    defer runner_result.deinit();
    var future = try ms.RunnerFuture.completed(allocator, &runner_result);
    return future.take_value();
}

fn array_from_owned(allocator: std.mem.Allocator, items: []const OwnedValue) !Array {
    const values = try allocator.alloc(Value, items.len);
    defer allocator.free(values);
    for (items, values) |*item, *value| value.* = item.borrow();
    return try Array.from_values(allocator, values);
}
