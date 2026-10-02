//! TVM MetaSchedule bindings.

const std = @import("std");
const ffi = @import("ffi.zig");
const container = @import("container.zig");
const tir = @import("tir.zig");

const Value = ffi.Value;
const Object = ffi.Object;

/// Workload stored in a MetaSchedule database.
pub const Workload = struct {
    object: Object,

    fn take(value: *ffi.OwnedValue) !Workload {
        try value.borrow().require_instance("meta_schedule.Workload");
        return .{ .object = try .take(value) };
    }

    fn as_value(self: *const Workload) Value {
        return self.object.as_value();
    }

    /// Release the TVM workload object.
    pub fn deinit(self: *Workload) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// One measured schedule stored by MetaSchedule.
pub const TuningRecord = struct {
    object: Object,

    fn take(value: *ffi.OwnedValue) !TuningRecord {
        try value.borrow().require_instance("meta_schedule.TuningRecord");
        return .{ .object = try .take(value) };
    }

    fn retain(value: Value) !TuningRecord {
        try value.require_instance("meta_schedule.TuningRecord");
        return .{ .object = try Object.retain(value) };
    }

    /// Copy the measured runtimes. The caller frees the returned slice.
    pub fn dupe_run_seconds(
        self: *const TuningRecord,
        /// Allocator used for the returned slice.
        allocator: std.mem.Allocator,
    ) ![]f64 {
        var run_seconds_value = try ffi.get_field(self.object.as_value(), "run_secs");
        defer run_seconds_value.deinit();
        if (run_seconds_value.borrow().is_none()) return error.MissingTimingResult;

        var run_seconds = try container.Array.retain(run_seconds_value.borrow());
        defer run_seconds.deinit();
        const count = try run_seconds.len(allocator);
        if (count == 0) return error.MissingTimingResult;

        const result = try allocator.alloc(f64, count);
        errdefer allocator.free(result);
        for (result, 0..) |*seconds, index| {
            var value = try run_seconds.get(allocator, index);
            defer value.deinit();
            seconds.* = try value.borrow().to_float();
            if (!std.math.isFinite(seconds.*) or seconds.* <= 0.0)
                return error.InvalidTimingResult;
        }
        return result;
    }

    /// Release the TVM tuning record.
    pub fn deinit(self: *TuningRecord) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// Ranked tuning records returned by a database query.
pub const TuningRecords = struct {
    values: container.Array,

    /// Number of records in the result.
    pub fn len(self: TuningRecords, allocator: std.mem.Allocator) !usize {
        return try self.values.len(allocator);
    }

    /// Return one record from the result.
    pub fn get(
        self: TuningRecords,
        allocator: std.mem.Allocator,
        index: usize,
    ) !TuningRecord {
        var value = try self.values.get(allocator, index);
        defer value.deinit();
        return try .retain(value.borrow());
    }

    /// Release the record array.
    pub fn deinit(self: *TuningRecords) void {
        self.values.deinit();
        self.* = undefined;
    }
};

/// TVM database used by MetaSchedule search and selection.
pub const Database = struct {
    object: Object,

    /// Take a packed result as a database.
    pub fn take(value: *ffi.OwnedValue) !Database {
        try value.borrow().require_instance("meta_schedule.Database");
        return .{ .object = try .take(value) };
    }

    /// Module comparison used to deduplicate workloads.
    pub const ModuleEquality = enum {
        structural,
        ignore_tensor,
        anchor_block,

        fn name(self: ModuleEquality) [:0]const u8 {
            return switch (self) {
                .structural => "structural",
                .ignore_tensor => "ignore-tensor",
                .anchor_block => "anchor-block",
            };
        }
    };

    /// Options for a file-backed JSON database.
    pub const JsonOptions = struct {
        /// Create absent database files.
        allow_missing: bool = true,
        /// Comparison used to deduplicate workload modules.
        module_equality: ModuleEquality = .structural,
    };

    /// Open TVM's file-backed JSON database.
    pub fn open_json(
        /// Allocator used by the packed call.
        allocator: std.mem.Allocator,
        /// File containing serialized workloads.
        workload_path: [:0]const u8,
        /// File containing serialized tuning records.
        record_path: [:0]const u8,
        /// File creation and module comparison policy.
        options: JsonOptions,
    ) !Database {
        return try ffi.call_global_take(
            Database,
            allocator,
            "meta_schedule.DatabaseJSONDatabase",
            &.{
                Value.str(workload_path),
                Value.str(record_path),
                Value.boolean(options.allow_missing),
                Value.str(options.module_equality.name()),
            },
        );
    }

    /// Return the workload corresponding to `module`, adding it when absent.
    pub fn commit_workload(
        self: *const Database,
        allocator: std.mem.Allocator,
        module: tir.IRModule,
    ) !Workload {
        var result = try ffi.call_global(
            allocator,
            "meta_schedule.DatabaseCommitWorkload",
            &.{ self.object.as_value(), module.as_value() },
        );
        errdefer result.deinit();
        return try .take(&result);
    }

    /// Return up to `count` records in database ranking order.
    pub fn top_k(
        self: *const Database,
        allocator: std.mem.Allocator,
        workload: *const Workload,
        count: usize,
    ) !TuningRecords {
        if (count == 0) return error.InvalidRecordCount;
        const packed_count = std.math.cast(i64, count) orelse
            return error.IntegerOutOfRange;
        var result = try ffi.call_global(
            allocator,
            "meta_schedule.DatabaseGetTopK",
            &.{ self.object.as_value(), workload.as_value(), Value.int(packed_count) },
        );
        defer result.deinit();
        return .{ .values = try container.Array.retain(result.borrow()) };
    }

    /// Apply the highest-ranked trace to `module`.
    pub fn query_module(
        self: *const Database,
        allocator: std.mem.Allocator,
        module: tir.IRModule,
        target: tir.Target,
        workload_name: [:0]const u8,
    ) !?tir.IRModule {
        var result = try ffi.call_global(
            allocator,
            "meta_schedule.DatabaseQueryIRModule",
            &.{
                self.object.as_value(),
                module.as_value(),
                target.as_value(),
                Value.str(workload_name),
            },
        );
        errdefer result.deinit();
        if (result.borrow().is_none()) {
            result.deinit();
            return null;
        }
        return try tir.IRModule.take(&result);
    }

    fn as_value(self: *const Database) Value {
        return self.object.as_value();
    }

    /// Release the TVM database object.
    pub fn deinit(self: *Database) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// MetaSchedule design-space generator.
pub const SpaceGenerator = struct {
    object: Object,

    /// Take a packed result as a design-space generator.
    pub fn take(value: *ffi.OwnedValue) !SpaceGenerator {
        try value.borrow().require_instance("meta_schedule.SpaceGenerator");
        return .{ .object = try .take(value) };
    }

    /// Create a generator using rules selected from the target.
    pub fn post_order_apply(allocator: std.mem.Allocator) !SpaceGenerator {
        return try ffi.call_global_take(
            SpaceGenerator,
            allocator,
            "meta_schedule.SpaceGeneratorPostOrderApply",
            &.{ Value.none(), Value.none(), Value.none(), Value.none() },
        );
    }

    /// Generate schedules for `module` after context initialization.
    pub fn generate(
        self: *const SpaceGenerator,
        allocator: std.mem.Allocator,
        module: tir.IRModule,
    ) !Schedules {
        var result = try ffi.call_global(
            allocator,
            "meta_schedule.SpaceGeneratorGenerateDesignSpace",
            &.{ self.object.as_value(), module.as_value() },
        );
        defer result.deinit();
        return .{ .values = try container.Array.retain(result.borrow()) };
    }

    fn as_value(self: *const SpaceGenerator) Value {
        return self.object.as_value();
    }

    /// Release the TVM space generator.
    pub fn deinit(self: *SpaceGenerator) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// MetaSchedule schedule search strategy.
pub const SearchStrategy = struct {
    object: Object,

    /// Take a packed result as a search strategy.
    pub fn take(value: *ffi.OwnedValue) !SearchStrategy {
        try value.borrow().require_instance("meta_schedule.SearchStrategy");
        return .{ .object = try .take(value) };
    }

    /// Options for randomized trace replay.
    pub const ReplayTraceOptions = struct {
        /// Maximum failed attempts while replaying one randomized trace.
        max_fail_count: i64 = 100,
    };

    /// Create a randomized trace-replay strategy.
    pub fn replay_trace(
        allocator: std.mem.Allocator,
        options: ReplayTraceOptions,
    ) !SearchStrategy {
        if (options.max_fail_count <= 0) return error.InvalidSearchConfiguration;
        return try ffi.call_global_take(
            SearchStrategy,
            allocator,
            "meta_schedule.SearchStrategyReplayTrace",
            &.{Value.int(options.max_fail_count)},
        );
    }

    fn as_value(self: *const SearchStrategy) Value {
        return self.object.as_value();
    }

    /// Release the TVM search strategy.
    pub fn deinit(self: *SearchStrategy) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// MetaSchedule task context.
pub const TuneContext = struct {
    object: Object,

    /// Take a packed result as a tuning context.
    pub fn take(value: *ffi.OwnedValue) !TuneContext {
        try value.borrow().require_instance("meta_schedule.TuneContext");
        return .{ .object = try .take(value) };
    }

    /// Inputs used to create a tuning context.
    pub const Options = struct {
        /// TIR module tuned by this task.
        module: tir.IRModule,
        /// Compilation target used to select schedule rules.
        target: tir.Target,
        /// Design-space generator.
        generator: *const SpaceGenerator,
        /// Search strategy.
        strategy: *const SearchStrategy,
        /// Task name recorded in tuning results.
        task_name: [:0]const u8,
        /// Number of threads exposed to the task context.
        num_threads: i64 = 1,
        /// Random seed used by schedule generation.
        random_seed: i64 = 42,
        /// Logging callback.
        logger: *const ffi.PackedFunction,
    };

    /// Create and initialize a tuning task.
    pub fn init(allocator: std.mem.Allocator, options: Options) !TuneContext {
        if (options.num_threads <= 0) return error.InvalidTaskConfiguration;
        var value = try ffi.call_global(allocator, "meta_schedule.TuneContext", &.{
            options.module.as_value(),
            options.target.as_value(),
            options.generator.as_value(),
            options.strategy.as_value(),
            Value.str(options.task_name),
            Value.int(options.num_threads),
            Value.int(options.random_seed),
            options.logger.as_value(),
        });
        errdefer value.deinit();
        try ffi.call_global_void(
            allocator,
            "meta_schedule.TuneContextInitialize",
            &.{value.borrow()},
        );
        return try .take(&value);
    }

    fn as_value(self: *const TuneContext) Value {
        return self.object.as_value();
    }

    /// Release the TVM task context.
    pub fn deinit(self: *TuneContext) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// MetaSchedule candidate compiler.
pub const Builder = struct {
    object: Object,

    /// Take a packed result as a candidate builder.
    pub fn take(value: *ffi.OwnedValue) !Builder {
        try value.borrow().require_instance("meta_schedule.Builder");
        return .{ .object = try .take(value) };
    }

    /// Create a builder backed by a packed callback.
    pub fn from_callback(
        allocator: std.mem.Allocator,
        callback: *const ffi.PackedFunction,
    ) !Builder {
        return try ffi.call_global_take(
            Builder,
            allocator,
            "meta_schedule.BuilderPyBuilder",
            &.{callback.as_value()},
        );
    }

    fn as_value(self: *const Builder) Value {
        return self.object.as_value();
    }

    /// Release the TVM builder.
    pub fn deinit(self: *Builder) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// One candidate submitted to a MetaSchedule builder callback.
pub const BuilderInput = struct {
    value: Value,

    /// Interpret a callback argument as a builder input.
    pub fn from_value(value: Value) !BuilderInput {
        try value.require_instance("meta_schedule.BuilderInput");
        return .{ .value = value };
    }

    /// Return the candidate's TIR module.
    pub fn module(self: BuilderInput) !tir.IRModule {
        var module_value = try ffi.get_field(self.value, "mod");
        defer module_value.deinit();
        return try tir.IRModule.retain(module_value.borrow());
    }
};

/// Result returned by a MetaSchedule builder callback.
pub const BuilderResult = struct {
    object: Object,

    /// Take a packed result as a builder result.
    pub fn take(value: *ffi.OwnedValue) !BuilderResult {
        try value.borrow().require_instance("meta_schedule.BuilderResult");
        return .{ .object = try .take(value) };
    }

    /// Candidate compilation outcome.
    pub const Outcome = union(enum) {
        /// Path to the compiled candidate artifact.
        success: [:0]const u8,
        /// Message describing a candidate compilation failure.
        failure: [:0]const u8,
    };

    /// Create a builder result from one outcome.
    pub fn init(allocator: std.mem.Allocator, outcome: Outcome) !BuilderResult {
        var artifact_path = Value.none();
        var error_message = Value.none();
        switch (outcome) {
            .success => |path| artifact_path = Value.str(path),
            .failure => |message| error_message = Value.str(message),
        }
        return try ffi.call_global_take(
            BuilderResult,
            allocator,
            "meta_schedule.BuilderResult",
            &.{ artifact_path, error_message },
        );
    }

    /// Move the packed result into a callback return array.
    pub fn take_value(self: *BuilderResult) ffi.OwnedValue {
        return self.object.take_value();
    }

    /// Release the TVM builder result.
    pub fn deinit(self: *BuilderResult) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// MetaSchedule candidate measurement runner.
pub const Runner = struct {
    object: Object,

    /// Take a packed result as a candidate runner.
    pub fn take(value: *ffi.OwnedValue) !Runner {
        try value.borrow().require_instance("meta_schedule.Runner");
        return .{ .object = try .take(value) };
    }

    /// Create a runner backed by a packed callback.
    pub fn from_callback(
        allocator: std.mem.Allocator,
        callback: *const ffi.PackedFunction,
    ) !Runner {
        return try ffi.call_global_take(
            Runner,
            allocator,
            "meta_schedule.RunnerPyRunner",
            &.{callback.as_value()},
        );
    }

    fn as_value(self: *const Runner) Value {
        return self.object.as_value();
    }

    /// Release the TVM runner.
    pub fn deinit(self: *Runner) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// One candidate submitted to a MetaSchedule runner callback.
pub const RunnerInput = struct {
    value: Value,

    /// Interpret a callback argument as a runner input.
    pub fn from_value(value: Value) !RunnerInput {
        try value.require_instance("meta_schedule.RunnerInput");
        return .{ .value = value };
    }

    /// Copy the candidate artifact path. The caller frees the returned bytes.
    pub fn dupe_artifact_path(self: RunnerInput, allocator: std.mem.Allocator) ![]u8 {
        var path_value = try ffi.get_field(self.value, "artifact_path");
        defer path_value.deinit();
        return try path_value.borrow().dupe_string(allocator);
    }
};

/// Completed measurement returned by a runner future.
pub const RunnerResult = struct {
    object: Object,

    /// Take a packed result as a runner result.
    pub fn take(value: *ffi.OwnedValue) !RunnerResult {
        try value.borrow().require_instance("meta_schedule.RunnerResult");
        return .{ .object = try .take(value) };
    }

    /// Candidate measurement outcome.
    pub const Outcome = union(enum) {
        /// Measured execution times in seconds.
        success: []const f64,
        /// Message describing a measurement failure.
        failure: [:0]const u8,
    };

    /// Create a runner result from one outcome.
    pub fn init(allocator: std.mem.Allocator, outcome: Outcome) !RunnerResult {
        var run_seconds: ?container.Array = null;
        defer if (run_seconds) |*values| values.deinit();

        var times = Value.none();
        var error_message = Value.none();
        switch (outcome) {
            .success => |samples| blk: {
                const values = try allocator.alloc(Value, samples.len);
                defer allocator.free(values);
                for (samples, values) |sample, *value| value.* = Value.float(sample);
                run_seconds = try container.Array.from_values(allocator, values);
                times = run_seconds.?.as_value();
                break :blk;
            },
            .failure => |message| error_message = Value.str(message),
        }
        return try ffi.call_global_take(
            RunnerResult,
            allocator,
            "meta_schedule.RunnerResult",
            &.{ times, error_message },
        );
    }

    fn as_value(self: *const RunnerResult) Value {
        return self.object.as_value();
    }

    /// Release the TVM runner result.
    pub fn deinit(self: *RunnerResult) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// MetaSchedule future containing a completed runner result.
pub const RunnerFuture = struct {
    object: Object,

    /// Take a packed result as a runner future.
    pub fn take(value: *ffi.OwnedValue) !RunnerFuture {
        try value.borrow().require_instance("meta_schedule.RunnerFuture");
        return .{ .object = try .take(value) };
    }

    /// Create a future whose result is already available.
    pub fn completed(
        allocator: std.mem.Allocator,
        result: *const RunnerResult,
    ) !RunnerFuture {
        const done_callback = struct {
            fn call(
                _: ?*anyopaque,
                args: *const ffi.CallbackArgs,
            ) !ffi.CallbackOutput {
                if (args.len() != 0) return error.InvalidCallbackArity;
                return .{ .boolean = true };
            }
        }.call;
        var done = try ffi.PackedFunction.create(null, done_callback, null);
        defer done.deinit();

        const ResultHolder = struct {
            result: ffi.OwnedValue,

            fn call(
                self_ptr: ?*anyopaque,
                args: *const ffi.CallbackArgs,
            ) !ffi.CallbackOutput {
                if (args.len() != 0) return error.InvalidCallbackArity;
                const self: *@This() = @ptrCast(@alignCast(self_ptr orelse
                    return error.MissingCallbackState));
                return .{ .owned = try ffi.OwnedValue.retain(self.result.borrow()) };
            }

            fn deinit(self_ptr: ?*anyopaque) void {
                const self: *@This() = @ptrCast(@alignCast(self_ptr orelse return));
                self.result.deinit();
                std.heap.c_allocator.destroy(self);
            }
        };
        const holder = try std.heap.c_allocator.create(ResultHolder);
        holder.result = ffi.OwnedValue.retain(result.as_value()) catch |err| {
            std.heap.c_allocator.destroy(holder);
            return err;
        };

        var get_result = ffi.PackedFunction.create(
            @ptrCast(holder),
            ResultHolder.call,
            ResultHolder.deinit,
        ) catch |err| {
            holder.result.deinit();
            std.heap.c_allocator.destroy(holder);
            return err;
        };
        defer get_result.deinit();

        return try ffi.call_global_take(
            RunnerFuture,
            allocator,
            "meta_schedule.RunnerFuture",
            &.{ done.as_value(), get_result.as_value() },
        );
    }

    /// Move the packed future into a callback return array.
    pub fn take_value(self: *RunnerFuture) ffi.OwnedValue {
        return self.object.take_value();
    }

    /// Release the TVM runner future.
    pub fn deinit(self: *RunnerFuture) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// Callback applied after MetaSchedule measures a candidate.
pub const MeasureCallback = struct {
    object: Object,

    /// Take a packed result as a measurement callback.
    pub fn take(value: *ffi.OwnedValue) !MeasureCallback {
        try value.borrow().require_instance("meta_schedule.MeasureCallback");
        return .{ .object = try .take(value) };
    }

    /// Create a callback that commits measurements to the tuning database.
    pub fn add_to_database(allocator: std.mem.Allocator) !MeasureCallback {
        return try ffi.call_global_take(
            MeasureCallback,
            allocator,
            "meta_schedule.MeasureCallbackAddToDatabase",
            &.{},
        );
    }

    fn as_value(self: *const MeasureCallback) Value {
        return self.object.as_value();
    }

    /// Release the TVM callback.
    pub fn deinit(self: *MeasureCallback) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// Scheduler that distributes trials among MetaSchedule tasks.
pub const TaskScheduler = struct {
    object: Object,

    /// Take a packed result as a task scheduler.
    pub fn take(value: *ffi.OwnedValue) !TaskScheduler {
        try value.borrow().require_instance("meta_schedule.TaskScheduler");
        return .{ .object = try .take(value) };
    }

    /// Inputs used to create a gradient-based task scheduler.
    pub const Options = struct {
        /// Logging callback.
        logger: *const ffi.PackedFunction,
        /// Gradient scheduler smoothing factor.
        alpha: f64 = 0.8,
        /// Recent-task window used by the scheduler.
        window_size: i64 = 3,
        /// Random seed used by task selection.
        random_seed: i64 = 42,
    };

    /// Create a gradient-based task scheduler.
    pub fn gradient_based(
        allocator: std.mem.Allocator,
        options: Options,
    ) !TaskScheduler {
        if (!std.math.isFinite(options.alpha)) return error.InvalidTaskConfiguration;
        if (options.alpha < 0.0) return error.InvalidTaskConfiguration;
        if (options.alpha > 1.0) return error.InvalidTaskConfiguration;
        if (options.window_size <= 0) return error.InvalidTaskConfiguration;
        return try ffi.call_global_take(
            TaskScheduler,
            allocator,
            "meta_schedule.TaskSchedulerGradientBased",
            &.{
                options.logger.as_value(),
                Value.float(options.alpha),
                Value.int(options.window_size),
                Value.int(options.random_seed),
            },
        );
    }

    /// Inputs and limits for one tuning loop.
    pub const TuneOptions = struct {
        /// Task contexts.
        contexts: []const *const TuneContext,
        /// Weight corresponding to each task context.
        weights: []const f64,
        /// Trial limit shared by all tasks.
        max_trials_global: i64,
        /// Per-task trial limit.
        max_trials_per_task: i64,
        /// Candidates requested from each iteration.
        trials_per_iter: i64,
        /// Candidate compiler.
        builder: *const Builder,
        /// Candidate runner.
        runner: *const Runner,
        /// Measurement callbacks.
        callbacks: []const *const MeasureCallback,
        /// Database receiving workloads and measurements.
        database: *const Database,
    };

    /// Run the tuning loop.
    pub fn tune(
        self: *const TaskScheduler,
        allocator: std.mem.Allocator,
        options: TuneOptions,
    ) !void {
        if (options.contexts.len == 0) return error.InvalidTaskConfiguration;
        if (options.contexts.len != options.weights.len)
            return error.InvalidTaskConfiguration;
        if (options.max_trials_global <= 0) return error.InvalidTaskConfiguration;
        if (options.max_trials_per_task <= 0) return error.InvalidTaskConfiguration;
        if (options.trials_per_iter <= 0) return error.InvalidTaskConfiguration;
        for (options.weights) |weight| {
            if (!std.math.isFinite(weight) or weight <= 0.0)
                return error.InvalidTaskConfiguration;
        }

        const context_values = try allocator.alloc(Value, options.contexts.len);
        defer allocator.free(context_values);
        for (options.contexts, context_values) |context, *value|
            value.* = context.as_value();
        var contexts = try container.Array.from_values(allocator, context_values);
        defer contexts.deinit();

        const weight_values = try allocator.alloc(Value, options.weights.len);
        defer allocator.free(weight_values);
        for (options.weights, weight_values) |weight, *value|
            value.* = Value.float(weight);
        var weights = try container.Array.from_values(allocator, weight_values);
        defer weights.deinit();

        const callback_values = try allocator.alloc(Value, options.callbacks.len);
        defer allocator.free(callback_values);
        for (options.callbacks, callback_values) |callback, *value|
            value.* = callback.as_value();
        var callbacks = try container.Array.from_values(allocator, callback_values);
        defer callbacks.deinit();

        // Parameter order follows TaskSchedulerNode::Tune in task_scheduler.h.
        try ffi.call_global_void(allocator, "meta_schedule.TaskSchedulerTune", &.{
            self.object.as_value(),
            contexts.as_value(),
            weights.as_value(),
            Value.int(options.max_trials_global),
            Value.int(options.max_trials_per_task),
            Value.int(options.trials_per_iter),
            options.builder.as_value(),
            options.runner.as_value(),
            callbacks.as_value(),
            options.database.as_value(),
            Value.none(),
        });
    }

    /// Release the TVM task scheduler.
    pub fn deinit(self: *TaskScheduler) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// Schedules generated for one TIR module.
pub const Schedules = struct {
    values: container.Array,

    /// Number of generated schedules.
    pub fn len(self: Schedules, allocator: std.mem.Allocator) !usize {
        return try self.values.len(allocator);
    }

    /// Return one generated schedule.
    pub fn get(self: Schedules, allocator: std.mem.Allocator, index: usize) !Schedule {
        var value = try self.values.get(allocator, index);
        defer value.deinit();
        try value.borrow().require_instance("tir.Schedule");
        return .{ .object = try Object.retain(value.borrow()) };
    }

    /// Release the schedule array.
    pub fn deinit(self: *Schedules) void {
        self.values.deinit();
        self.* = undefined;
    }
};

/// One generated TIR schedule.
pub const Schedule = struct {
    object: Object,

    /// Render the schedule trace as Python source lines.
    pub fn python_trace(
        self: *const Schedule,
        allocator: std.mem.Allocator,
    ) !PythonTrace {
        var trace = try ffi.call_global(
            allocator,
            "tir.schedule.ScheduleGetTrace",
            &.{self.object.as_value()},
        );
        defer trace.deinit();
        var lines_value = try ffi.call_global(
            allocator,
            "tir.schedule.TraceAsPython",
            &.{ trace.borrow(), Value.boolean(false) },
        );
        defer lines_value.deinit();
        var lines = try container.Array.retain(lines_value.borrow());
        defer lines.deinit();

        const count = try lines.len(allocator);
        const result = try allocator.alloc([]u8, count);
        var initialized: usize = 0;
        errdefer {
            for (result[0..initialized]) |line| allocator.free(line);
            allocator.free(result);
        }
        for (result, 0..) |*line, index| {
            var value = try lines.get(allocator, index);
            defer value.deinit();
            line.* = try value.borrow().dupe_string(allocator);
            initialized += 1;
        }
        return .{ .allocator = allocator, .lines = result };
    }

    /// Release the TVM schedule object.
    pub fn deinit(self: *Schedule) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// Python rendering of a TIR schedule trace.
pub const PythonTrace = struct {
    allocator: std.mem.Allocator,
    lines: [][]u8,

    /// Release the rendered lines.
    pub fn deinit(self: *PythonTrace) void {
        for (self.lines) |line| self.allocator.free(line);
        self.allocator.free(self.lines);
        self.* = undefined;
    }
};

/// Count the default CUDA tensor-core schedule rules available to MetaSchedule.
pub fn default_cuda_tensor_core_rule_count(allocator: std.mem.Allocator) !usize {
    var rules_value = try ffi.call_global(
        allocator,
        "meta_schedule.ScheduleRuleDefaultCUDATensorCore",
        &.{},
    );
    defer rules_value.deinit();
    var rules = try container.Array.retain(rules_value.borrow());
    defer rules.deinit();
    return try rules.len(allocator);
}

/// Register the callback used by MetaSchedule to query host CPU count.
pub fn register_cpu_count_callback(function: *const ffi.PackedFunction) !void {
    try ffi.set_global("meta_schedule._cpu_count", function, true);
    try ffi.set_global("meta_schedule.cpu_count", function, true);
}
