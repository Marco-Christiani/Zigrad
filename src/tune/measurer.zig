//! Provider-candidate evaluation contracts and executable profitability policy.

const std = @import("std");

const device_mod = @import("device");
const Executor = @import("../execution.zig");
const kernel = @import("kernel");
const pr = @import("pr");
const HostBuffer = @import("../utils.zig").HostBuffer;

const log = std.log.scoped(.@"zg/tune_measurer");

/// One paired observation of an unreplaced and provider implementation.
pub const Sample = struct {
    /// Latency of the unreplaced callable.
    unreplaced_ns: u64,
    /// Latency of the provider implementation.
    selected_ns: u64,
};

/// Measurement of one provider implementation against its unreplaced callable.
pub const Measurement = union(enum) {
    /// No valid measurement is available.
    unavailable,
    /// Execution completed but produced different outputs.
    incorrect,
    /// Comparable paired samples that passed output validation.
    measured: []const Sample,

    /// Release sample storage owned by this result.
    pub fn deinit(self: Measurement, allocator: std.mem.Allocator) void {
        switch (self) {
            .measured => |samples| allocator.free(samples),
            .unavailable, .incorrect => {},
        }
    }
};

/// Release a measurement slice and every owned sample allocation.
pub fn deinit_measurements(
    /// Allocator that owns the result slice and measured samples.
    allocator: std.mem.Allocator,
    /// Measurement result returned by a `Measurer`.
    measurements: []const Measurement,
) void {
    for (measurements) |result| result.deinit(allocator);
    allocator.free(measurements);
}

/// Failures produced while comparing executable candidates.
pub const Error = error{
    /// Integration preparation, execution, or result collection failed.
    MeasurementFailed,
} || std.mem.Allocator.Error;

/// Type-erased measurement of provider artifacts against one callable.
pub const Measurer = struct {
    /// Measurement state borrowed for the lifetime of this interface.
    context: *anyopaque,
    /// Static dispatch table for `context`.
    vtable: *const VTable,

    pub const VTable = struct {
        /// Return one measurement per candidate in candidate order.
        measure: *const fn (
            context: *anyopaque,
            func: pr.Function,
            candidates: []const kernel.ProviderCandidate,
            selected_device: device_mod.Device,
            allocator: std.mem.Allocator,
        ) Error![]Measurement,
    };

    /// Measure every provider implementation of one callable.
    ///
    /// The result contains one entry per candidate in the same order. Caller
    ///  releases it with `deinit_measurements`.
    pub fn measure(
        self: Measurer,
        /// Callable shared by the unreplaced and provider implementations.
        func: pr.Function,
        /// Compiled provider implementations to measure.
        candidates: []const kernel.ProviderCandidate,
        /// Device targeted by every implementation.
        selected_device: device_mod.Device,
        /// Allocator owning the returned measurement result.
        allocator: std.mem.Allocator,
    ) Error![]Measurement {
        return try self.vtable.measure(
            self.context,
            func,
            candidates,
            selected_device,
            allocator,
        );
    }
};

/// Loaded program and integration state owned by an executable factory.
pub const Executable = struct {
    /// Loaded callable and the executor that owns its handle.
    program: Executor.LoadedProgram,
    /// Factory state passed to `deinit_fn`.
    state: *anyopaque,
    /// Releases `program` and `state` together.
    deinit_fn: *const fn (state: *anyopaque, program: *Executor.LoadedProgram) void,

    /// Release the loaded program and its integration-owned state.
    pub fn deinit(self: *Executable) void {
        self.deinit_fn(self.state, &self.program);
        self.* = undefined;
    }
};

/// Integration-specific preparation of callable implementations.
pub const ExecutableFactory = struct {
    /// Factory state borrowed for the lifetime of this interface.
    context: *anyopaque,
    /// Static dispatch table for `context`.
    vtable: *const VTable,

    pub const VTable = struct {
        /// Return the device used by prepared executables.
        device: *const fn (context: *anyopaque) device_mod.Device,
        /// Prepare one callable implementation for execution.
        prepare: *const fn (
            context: *anyopaque,
            func: pr.Function,
            implementation: kernel.Implementation,
            allocator: std.mem.Allocator,
        ) Error!Executable,
    };

    /// Return the device targeted by this factory.
    pub fn device(self: ExecutableFactory) device_mod.Device {
        return self.vtable.device(self.context);
    }

    /// Prepare one implementation on the factory device.
    pub fn prepare(
        self: ExecutableFactory,
        /// Callable whose implementation is prepared.
        func: pr.Function,
        /// Unreplaced or provider implementation to prepare.
        implementation: kernel.Implementation,
        /// Allocator retained by the returned executable when needed.
        allocator: std.mem.Allocator,
    ) Error!Executable {
        var executable = try self.vtable.prepare(
            self.context,
            func,
            implementation,
            allocator,
        );
        if (!executable.program.executor.device.eql(self.device())) {
            executable.deinit();
            return error.MeasurementFailed;
        }
        return executable;
    }
};

/// Supplies parameter values used for correctness checks and timing samples.
pub const InputGenerator = struct {
    /// Generator state borrowed for the lifetime of this interface.
    context: *anyopaque,
    /// Static dispatch table for `context`.
    vtable: *const VTable,

    pub const VTable = struct {
        /// Fill one host buffer for a callable parameter.
        fill: *const fn (
            context: *anyopaque,
            func: pr.Function,
            param_index: usize,
            destination: []u8,
        ) Error!void,
    };

    /// Fill one callable parameter in its native byte representation.
    pub fn fill(
        self: InputGenerator,
        /// Callable whose parameter is being initialized.
        func: pr.Function,
        /// Parameter position corresponding to `destination`.
        param_index: usize,
        /// Exact native storage for the parameter dtype and shape. The
        ///  implementation initializes every byte.
        destination: []u8,
    ) Error!void {
        return try self.vtable.fill(self.context, func, param_index, destination);
    }
};

/// Measures an unreplaced callable and provider artifacts through one factory.
///
/// Inputs remain resident and unchanged across warmups and paired samples. Each
///  invocation receives fresh output buffers.
pub const ExecutableMeasurer = struct {
    /// Measurement policy.
    pub const Options = struct {
        /// Untimed invocations of each implementation before sampling.
        warmups: usize = 3,
        /// Paired samples collected with alternating execution order.
        ///
        /// The valid range is 1 through 63.
        samples: usize = 15,
        /// Absolute tolerance used for floating-point output comparison.
        absolute_tolerance: f64 = 1e-4,
        /// Relative tolerance used for floating-point output comparison.
        relative_tolerance: f64 = 1e-4,
        /// Optional source of callable inputs. The default fills deterministic
        ///  values derived from each element position.
        input_generator: ?InputGenerator = null,
    };

    /// I/O state used for monotonic timestamps and provider operations.
    io: std.Io,
    /// Integration used to prepare every executable alternative.
    factory: ExecutableFactory,
    /// Correctness and sampling policy.
    options: Options = .{},

    /// Return the measurement interface consumed by candidate collection.
    pub fn interface(self: *ExecutableMeasurer) Measurer {
        return .{
            .context = @ptrCast(self),
            .vtable = &.{ .measure = measure },
        };
    }

    fn measure(
        ptr: *anyopaque,
        func: pr.Function,
        candidates: []const kernel.ProviderCandidate,
        selected_device: device_mod.Device,
        allocator: std.mem.Allocator,
    ) Error![]Measurement {
        const self: *ExecutableMeasurer = @ptrCast(@alignCast(ptr));
        return self.measure_impl(func, candidates, selected_device, allocator) catch |err| {
            log.err("candidate measurement failed for '{s}': {s}", .{ func.name, @errorName(err) });
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.MeasurementFailed,
            };
        };
    }

    fn measure_impl(
        self: *ExecutableMeasurer,
        func: pr.Function,
        candidates: []const kernel.ProviderCandidate,
        selected_device: device_mod.Device,
        allocator: std.mem.Allocator,
    ) ![]Measurement {
        try validate_options(self.options);
        if (!selected_device.eql(self.factory.device())) {
            return error.DeviceMismatch;
        }
        if (function_may_have_side_effects(func)) {
            return error.SideEffectingFunction;
        }

        const host_inputs = try make_inputs(
            allocator,
            func,
            self.options.input_generator,
        );
        defer free_host_buffers(allocator, host_inputs);

        var unreplaced: Prepared = undefined;
        try unreplaced.init(
            allocator,
            self.factory,
            func,
            .unreplaced,
            host_inputs,
            func.returns.len,
        );
        defer unreplaced.deinit();

        const expected = try unreplaced.capture(self.io, allocator, func.returns);
        defer free_host_buffers(allocator, expected);

        const measurements = try allocator.alloc(Measurement, candidates.len);
        @memset(measurements, .unavailable);
        errdefer deinit_measurements(allocator, measurements);
        for (candidates, 0..) |candidate, candidate_index| {
            measurements[candidate_index] = self.measure_candidate(
                allocator,
                func,
                &unreplaced,
                host_inputs,
                expected,
                candidate,
            ) catch |err| {
                if (err == error.OutOfMemory) return error.OutOfMemory;
                log.warn("candidate '{s}' could not be measured: {s}", .{
                    candidate.provider_name,
                    @errorName(err),
                });
                continue;
            };
        }
        return measurements;
    }

    fn measure_candidate(
        self: *ExecutableMeasurer,
        allocator: std.mem.Allocator,
        func: pr.Function,
        unreplaced: *Prepared,
        host_inputs: []const HostBuffer,
        expected: []const HostBuffer,
        candidate: kernel.ProviderCandidate,
    ) !Measurement {
        var prepared: Prepared = undefined;
        try prepared.init(
            allocator,
            self.factory,
            func,
            .{ .provider = candidate },
            host_inputs,
            func.returns.len,
        );
        defer prepared.deinit();

        const actual = try prepared.capture(self.io, allocator, func.returns);
        defer free_host_buffers(allocator, actual);
        if (!outputs_close(expected, actual, self.options)) {
            log.warn("candidate '{s}' failed output comparison", .{candidate.provider_name});
            return .incorrect;
        }

        for (0..self.options.warmups) |_| {
            _ = try self.run_pair(unreplaced, &prepared, false);
        }

        const samples = try allocator.alloc(Sample, self.options.samples);
        errdefer allocator.free(samples);

        for (samples, 0..) |*sample, sample_index| {
            sample.* = try self.run_pair(
                unreplaced,
                &prepared,
                sample_index % 2 == 1,
            );
        }

        log.info("candidate '{s}': collected {d} paired samples", .{
            candidate.provider_name,
            samples.len,
        });
        return .{ .measured = samples };
    }

    fn run_pair(
        self: *ExecutableMeasurer,
        unreplaced: *Prepared,
        candidate: *Prepared,
        candidate_first: bool,
    ) !Sample {
        if (candidate_first) {
            const measured_candidate = try candidate.measure(self.io);
            const measured_unreplaced = try unreplaced.measure(self.io);
            return .{
                .unreplaced_ns = measured_unreplaced,
                .selected_ns = measured_candidate,
            };
        } else {
            const measured_unreplaced = try unreplaced.measure(self.io);
            const measured_candidate = try candidate.measure(self.io);
            return .{
                .unreplaced_ns = measured_unreplaced,
                .selected_ns = measured_candidate,
            };
        }
    }
};

const Prepared = struct {
    allocator: std.mem.Allocator,
    executable: Executable,
    inputs: []Executor.Buffer,
    output_count: usize,

    fn init(
        self: *Prepared,
        allocator: std.mem.Allocator,
        factory: ExecutableFactory,
        func: pr.Function,
        implementation: kernel.Implementation,
        host_inputs: []const HostBuffer,
        output_count: usize,
    ) !void {
        self.allocator = allocator;
        self.executable = try factory.prepare(func, implementation, allocator);
        errdefer self.executable.deinit();

        self.inputs = try allocator.alloc(Executor.Buffer, host_inputs.len);
        errdefer allocator.free(self.inputs);
        var input_count: usize = 0;
        errdefer for (self.inputs[0..input_count]) |input|
            self.executable.program.executor.release(input);
        for (host_inputs, self.inputs) |input, *buffer| {
            buffer.* = try self.executable.program.executor.upload(
                input.data(),
                input.dtype,
                input.shape.const_slice(),
            );
            input_count += 1;
        }
        self.output_count = output_count;
    }

    fn deinit(self: *Prepared) void {
        for (self.inputs) |input| self.executable.program.executor.release(input);
        self.allocator.free(self.inputs);
        self.executable.deinit();
        self.* = undefined;
    }

    fn measure(self: *Prepared, io: std.Io) !u64 {
        var run = try self.invoke(io);
        defer run.deinit(self.executable.program.executor, self.allocator);
        return run.elapsed_ns;
    }

    fn capture(
        self: *Prepared,
        io: std.Io,
        allocator: std.mem.Allocator,
        output_avals: []const *pr.Var,
    ) ![]HostBuffer {
        var run = try self.invoke(io);
        defer run.deinit(self.executable.program.executor, self.allocator);

        const outputs = try allocator.alloc(HostBuffer, output_avals.len);
        var output_count: usize = 0;
        errdefer {
            for (outputs[0..output_count]) |*output| output.deinit();
            allocator.free(outputs);
        }
        for (output_avals, run.outputs, outputs) |output, buffer, *host| {
            const tensor = output.as_tensor();
            host.* = try HostBuffer.init(
                allocator,
                .from_slice(tensor.shape.dims),
                tensor.dtype,
            );
            errdefer host.deinit();
            if (try self.executable.program.executor.download(buffer, host.data_mut())) |event| {
                defer self.executable.program.executor.release_event(event);
                try self.executable.program.executor.wait(event);
            }
            output_count += 1;
        }
        return outputs;
    }

    fn invoke(self: *Prepared, io: std.Io) !Run {
        const outputs = try self.allocator.alloc(Executor.Buffer, self.output_count);
        errdefer self.allocator.free(outputs);
        const start = std.Io.Timestamp.now(io, .awake);
        const event = try self.executable.program.executor.invoke(
            self.executable.program,
            self.inputs,
            outputs,
            .{},
        );
        // A successful invocation initializes every output slot.
        errdefer for (outputs) |output| self.executable.program.executor.release(output);
        if (event) |completion| {
            defer self.executable.program.executor.release_event(completion);
            try self.executable.program.executor.wait(completion);
        }
        return .{
            .outputs = outputs,
            .elapsed_ns = @intCast(start.untilNow(io, .awake).toNanoseconds()),
        };
    }
};

const Run = struct {
    outputs: []Executor.Buffer,
    elapsed_ns: u64,

    fn deinit(
        self: *Run,
        executor: *Executor,
        allocator: std.mem.Allocator,
    ) void {
        for (self.outputs) |output| executor.release(output);
        allocator.free(self.outputs);
        self.* = undefined;
    }
};

fn make_inputs(
    allocator: std.mem.Allocator,
    func: pr.Function,
    generator: ?InputGenerator,
) ![]HostBuffer {
    const result = try allocator.alloc(HostBuffer, func.params.len);
    var count: usize = 0;
    errdefer {
        for (result[0..count]) |*input| input.deinit();
        allocator.free(result);
    }
    for (func.params, result, 0..) |param, *input, param_index| {
        const tensor = param.as_tensor();
        input.* = try HostBuffer.init(
            allocator,
            .from_slice(tensor.shape.dims),
            tensor.dtype,
        );
        errdefer input.deinit();
        if (generator) |selected| {
            try selected.fill(func, param_index, input.data_mut());
        } else {
            fill_input(input.data_mut(), tensor.dtype);
        }
        count += 1;
    }
    return result;
}

fn free_host_buffers(allocator: std.mem.Allocator, buffers: []HostBuffer) void {
    for (buffers) |*buffer| buffer.deinit();
    allocator.free(buffers);
}

fn fill_input(bytes: []u8, dtype: pr.DType) void {
    const width = dtype.size_in_bytes();
    for (0..bytes.len / width) |index| {
        const value = @as(f64, @floatFromInt(@as(i32, @intCast(index % 23)) - 11)) / 16.0;
        const destination = bytes[index * width ..][0..width];
        switch (dtype) {
            .f16 => write_value(u16, destination, pr.DType.f16.encode(f64, value)),
            .bf16 => write_value(u16, destination, pr.DType.bf16.encode(f64, value)),
            .f32 => write_value(f32, destination, @floatCast(value)),
            .f64 => write_value(f64, destination, value),
            .i8 => write_value(i8, destination, @intCast(@as(i32, @intCast(index % 23)) - 11)),
            .u8 => write_value(u8, destination, @intCast(index % 23)),
            .i32 => write_value(i32, destination, @intCast(@as(i32, @intCast(index % 23)) - 11)),
            .i64 => write_value(i64, destination, @intCast(@as(i32, @intCast(index % 23)) - 11)),
            .u32 => write_value(u32, destination, @intCast(index % 23)),
            .u64 => write_value(u64, destination, @intCast(index % 23)),
            .bool => destination[0] = @intFromBool(index % 2 == 0),
        }
    }
}

fn write_value(comptime T: type, destination: []u8, value: T) void {
    std.debug.assert(destination.len == @sizeOf(T));
    @memcpy(destination, std.mem.asBytes(&value));
}

fn outputs_close(
    expected: []const HostBuffer,
    actual: []const HostBuffer,
    options: ExecutableMeasurer.Options,
) bool {
    if (expected.len != actual.len) return false;
    for (expected, actual) |reference, candidate| {
        if (reference.dtype != candidate.dtype or
            !std.mem.eql(i64, reference.shape.const_slice(), candidate.shape.const_slice()) or
            reference.data().len != candidate.data().len)
        {
            return false;
        }
        if (!tensor_close(reference, candidate, options)) return false;
    }
    return true;
}

fn tensor_close(
    expected: HostBuffer,
    actual: HostBuffer,
    options: ExecutableMeasurer.Options,
) bool {
    return switch (expected.dtype) {
        .f16, .bf16, .f32, .f64 => floats_close(expected, actual, options),
        else => std.mem.eql(u8, expected.data(), actual.data()),
    };
}

fn floats_close(
    expected: HostBuffer,
    actual: HostBuffer,
    options: ExecutableMeasurer.Options,
) bool {
    const width = expected.dtype.size_in_bytes();
    const expected_bytes = expected.data();
    const actual_bytes = actual.data();
    for (0..expected_bytes.len / width) |index| {
        const offset = index * width;
        const reference: f64 = switch (expected.dtype) {
            .f16 => pr.DType.f16.decode(f64, std.mem.bytesToValue(u16, expected_bytes[offset..][0..2])),
            .bf16 => pr.DType.bf16.decode(f64, std.mem.bytesToValue(u16, expected_bytes[offset..][0..2])),
            .f32 => std.mem.bytesToValue(f32, expected_bytes[offset..][0..4]),
            .f64 => std.mem.bytesToValue(f64, expected_bytes[offset..][0..8]),
            else => unreachable,
        };
        const candidate: f64 = switch (actual.dtype) {
            .f16 => pr.DType.f16.decode(f64, std.mem.bytesToValue(u16, actual_bytes[offset..][0..2])),
            .bf16 => pr.DType.bf16.decode(f64, std.mem.bytesToValue(u16, actual_bytes[offset..][0..2])),
            .f32 => std.mem.bytesToValue(f32, actual_bytes[offset..][0..4]),
            .f64 => std.mem.bytesToValue(f64, actual_bytes[offset..][0..8]),
            else => unreachable,
        };
        if (std.math.isNan(reference) or std.math.isNan(candidate)) return false;
        if (std.math.isInf(reference) or std.math.isInf(candidate)) {
            if (reference != candidate) return false;
            continue;
        }
        const tolerance = options.absolute_tolerance +
            options.relative_tolerance * @abs(reference);
        if (@abs(candidate - reference) > tolerance) return false;
    }
    return true;
}

fn validate_options(options: ExecutableMeasurer.Options) !void {
    if (options.samples == 0 or options.samples > 63 or
        options.absolute_tolerance < 0.0 or options.relative_tolerance < 0.0)
    {
        return error.InvalidOptions;
    }
}

fn function_may_have_side_effects(func: pr.Function) bool {
    for (func.ops) |op| switch (op.params) {
        .custom_call => |params| if (params.has_side_effect) return true,
        // Calls require the source program for recursive effect analysis.
        .call => return true,
        else => {},
    };
    return false;
}
