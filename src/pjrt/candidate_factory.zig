//! PJRT preparation of callable implementations for profitability evaluation.

const std = @import("std");

const device_mod = @import("device");
const kernel = @import("kernel");
const mlir = @import("../mlir.zig");
const fingerprint = @import("pr").analysis.fingerprint;
const pr = @import("pr");
const measurer = @import("../tune/measurer.zig");
const client_mod = @import("client.zig");
const Execution = @import("execution.zig").Execution;
const LoadedProgram = @import("../execution.zig").LoadedProgram;

const log = std.log.scoped(.@"zg/pjrt_candidate_factory");

/// Prepares unreplaced callables and dispatchable provider artifacts through PJRT.
pub const CandidateExecutableFactory = struct {
    /// PJRT client and device used for every prepared executable.
    execution: *Execution,
    /// Providers whose artifacts may be dispatched by candidate executables.
    providers: []const kernel.KernelProvider,
    /// PJRT compilation options applied to every implementation.
    compile_options: client_mod.CompileOptions = .{},

    /// Return the executable-factory interface used by candidate measurement.
    pub fn interface(self: *CandidateExecutableFactory) measurer.ExecutableFactory {
        return .{
            .context = @ptrCast(self),
            .vtable = &.{
                .device = device,
                .prepare = prepare,
            },
        };
    }

    fn device(ptr: *anyopaque) device_mod.Device {
        const self: *CandidateExecutableFactory = @ptrCast(@alignCast(ptr));
        return self.execution.interface.device;
    }

    fn prepare(
        ptr: *anyopaque,
        func: pr.Function,
        implementation: kernel.Implementation,
        allocator: std.mem.Allocator,
    ) measurer.Error!measurer.Executable {
        const self: *CandidateExecutableFactory = @ptrCast(@alignCast(ptr));
        return self.prepare_impl(func, implementation, allocator) catch |err| {
            log.err("PJRT candidate preparation failed for '{s}': {s}", .{
                func.name,
                @errorName(err),
            });
            return switch (err) {
                error.OutOfMemory => error.OutOfMemory,
                else => error.MeasurementFailed,
            };
        };
    }

    fn prepare_impl(
        self: *CandidateExecutableFactory,
        func: pr.Function,
        implementation: kernel.Implementation,
        allocator: std.mem.Allocator,
    ) !measurer.Executable {
        try kernel.validate_providers(self.providers);
        return switch (implementation) {
            .unreplaced => try self.prepare_unreplaced(func, allocator),
            .provider => |candidate| try self.prepare_provider(func, candidate, allocator),
        };
    }

    fn prepare_unreplaced(
        self: *CandidateExecutableFactory,
        func: pr.Function,
        allocator: std.mem.Allocator,
    ) !measurer.Executable {
        const bytes = try lower_callable(allocator, func);
        defer allocator.free(bytes);

        const state = try allocator.create(UnreplacedState);
        errdefer allocator.destroy(state);
        state.* = .{
            .allocator = allocator,
            .execution = try Execution.init(
                self.execution.client,
                self.execution.device,
                .{},
            ),
        };
        const program = try compile_loaded(
            &state.execution,
            bytes,
            self.compile_options,
        );
        return .{
            .program = program,
            .state = state,
            .deinit_fn = UnreplacedState.deinit,
        };
    }

    fn prepare_provider(
        self: *CandidateExecutableFactory,
        func: pr.Function,
        candidate: kernel.ProviderCandidate,
        allocator: std.mem.Allocator,
    ) !measurer.Executable {
        _ = kernel.find_provider(self.providers, candidate.provider_name) orelse
            return error.ProviderNotConfigured;

        const function_fingerprint = try fingerprint.function(allocator, func);
        const selection_key = try kernel.make_implementation_key(
            allocator,
            .{ .one = candidate.provider_name },
            self.execution.interface.device,
            function_fingerprint,
        );
        defer allocator.free(selection_key.bytes);

        const state = try allocator.create(ProviderState);
        errdefer allocator.destroy(state);
        state.allocator = allocator;
        state.store = kernel.KernelStore.init(allocator);
        errdefer state.store.deinit();
        state.registry = kernel.DispatchRegistry.init(allocator);
        errdefer state.registry.deinit();

        try state.store.put(selection_key, .{
            .candidate = .{ .provider = .{
                .provider_name = candidate.provider_name,
                .artifact = .{
                    .data = try allocator.dupe(u8, candidate.artifact.data),
                    .workspace_bytes = candidate.artifact.workspace_bytes,
                    .workspace_alignment = candidate.artifact.workspace_alignment,
                },
            } },
            .reason = "candidate under measurement",
        });
        try state.registry.register_providers(self.providers);
        try state.registry.prepare(&state.store, .{
            .device = self.execution.interface.device,
        });

        var candidate_program = pr.Program.init(allocator);
        defer candidate_program.deinit();
        const candidate_func = try build_dispatch_function(
            &candidate_program,
            func,
            selection_key.bytes,
        );
        const bytes = try lower_callable(allocator, candidate_func);
        defer allocator.free(bytes);

        state.execution = try Execution.init(
            self.execution.client,
            self.execution.device,
            .{
                .store = &state.store,
                .dispatch_registry = &state.registry,
            },
        );
        const program = try compile_loaded(
            &state.execution,
            bytes,
            self.compile_options,
        );
        return .{
            .program = program,
            .state = state,
            .deinit_fn = ProviderState.deinit,
        };
    }
};

const UnreplacedState = struct {
    allocator: std.mem.Allocator,
    execution: Execution,

    fn deinit(ptr: *anyopaque, program: *LoadedProgram) void {
        const self: *UnreplacedState = @ptrCast(@alignCast(ptr));
        program.deinit();
        self.allocator.destroy(self);
    }
};

const ProviderState = struct {
    allocator: std.mem.Allocator,
    store: kernel.KernelStore,
    registry: kernel.DispatchRegistry,
    execution: Execution,

    fn deinit(ptr: *anyopaque, program: *LoadedProgram) void {
        const self: *ProviderState = @ptrCast(@alignCast(ptr));
        program.deinit();
        self.registry.deinit();
        self.store.deinit();
        self.allocator.destroy(self);
    }
};

fn lower_callable(allocator: std.mem.Allocator, func: pr.Function) ![]u8 {
    return try mlir.stablehlo.lower_function_to_mlir(
        allocator,
        func,
        .mlir_bytecode,
    );
}

fn compile_loaded(
    execution: *Execution,
    stablehlo_bytes: []const u8,
    options: client_mod.CompileOptions,
) !LoadedProgram {
    var executable = try execution.client.compile(
        &execution.device,
        stablehlo_bytes,
        .binary,
        options,
    );
    const handle = execution.adopt_handle(executable) catch |err| {
        execution.client.deinit_executable(&executable);
        return err;
    };
    return .{
        .executor = &execution.interface,
        .handle = handle,
    };
}

fn build_dispatch_function(
    program: *pr.Program,
    source: pr.Function,
    selection_key: []const u8,
) !pr.Function {
    const allocator = program.allocator();
    var builder = try pr.FunctionBuilder.init(program, "candidate");
    defer builder.deinit();

    const inputs = try allocator.alloc(*pr.Var, source.params.len);
    defer allocator.free(inputs);
    for (source.params, inputs) |param, *input|
        input.* = try builder.param_like(param.aval);

    const output_avals = try allocator.alloc(pr.Aval, source.returns.len);
    defer allocator.free(output_avals);
    for (source.returns, output_avals) |output, *aval| aval.* = output.aval;

    const call = try builder.custom_call(.{
        .target_name = kernel.dispatch_target_name,
        .has_side_effect = false,
        .payload = selection_key,
    }, inputs, output_avals);
    return try builder.finish(.{ .returns = call.outputs });
}
