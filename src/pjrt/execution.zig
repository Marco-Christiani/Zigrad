//! PJRT implementation of the framework execution contract.

const std = @import("std");

const pjrt_api = @import("../c/pjrt/api.zig");
const Executor = @import("../execution.zig");
const client_mod = @import("client.zig");

const log = std.log.scoped(.@"zg/pjrt_execution");

const IntegrationError = pjrt_api.Error || error{
    FunctionNotAvailable,
    TypedFfiUnavailable,
    TypedFfiRegistrationFailed,
    PjrtReturnedNullBuffer,
    PjrtReturnedNullEvent,
    PjrtReturnedNullExecuteContext,
    OutputArityMismatch,
    PjrtReturnedNullPlatformName,
};

/// One PJRT client, selected device, and invocation configuration.
pub const Execution = struct {
    interface: Executor,
    client: *client_mod.Client,
    device: client_mod.Device,
    dispatch: client_mod.DispatchOptions,

    /// Configure execution on one PJRT device.
    pub fn init(
        client: *client_mod.Client,
        device: client_mod.Device,
        dispatch: client_mod.DispatchOptions,
    ) Executor.Error!Execution {
        if (dispatch.store != null or dispatch.dispatch_registry != null) {
            client.register_kernel_dispatcher() catch |err| return map_error(err);
        }

        const ordinal = device.get_local_hardware_id(client.api) catch |err|
            return map_error(err);

        return .{
            .interface = .{
                .device = .{
                    .platform = client.platform,
                    .ordinal = ordinal,
                },
                .vtable = &vtable,
            },
            .client = client,
            .device = device,
            .dispatch = dispatch,
        };
    }

    /// Transfer a native PJRT executable into a framework program handle.
    pub fn adopt_handle(
        self: *Execution,
        loaded_executable: client_mod.LoadedExecutable,
    ) Executor.Error!*anyopaque {
        const executable = try self.client.allocator.create(client_mod.LoadedExecutable);
        executable.* = loaded_executable;
        return @ptrCast(executable);
    }

    /// Access the PJRT executable represented by a loaded program.
    pub fn loaded(
        self: *const Execution,
        program: Executor.LoadedProgram,
    ) Executor.Error!*client_mod.LoadedExecutable {
        if (program.executor != &self.interface) return error.InvalidArgument;
        return loaded_handle(program.handle);
    }

    const vtable: Executor.VTable = .{
        .upload = upload,
        .download = download,
        .invoke = invoke,
        .await_event = await_event,
        .release_buffer = release_buffer,
        .release_event = release_event,
        .release_program = release_program,
    };

    fn promote(interface: *Executor) *Execution {
        return @fieldParentPtr("interface", interface);
    }

    fn upload(
        interface: *Executor,
        data: []const u8,
        dtype: @import("dtype").DType,
        shape: []const i64,
    ) Executor.Error!Executor.Buffer {
        const self = promote(interface);
        const buffer = self.client.buffer_from_host(
            &self.device,
            data,
            dtype,
            shape,
        ) catch |err| return map_error(err);
        return wrap_buffer(buffer);
    }

    fn download(
        interface: *Executor,
        buffer: Executor.Buffer,
        destination: []u8,
    ) Executor.Error!?Executor.Event {
        const self = promote(interface);
        var pjrt_buffer = unwrap_buffer(buffer);

        if (pjrt_buffer.is_on_cpu(self.client.api) catch false) {
            const source_address = pjrt_buffer.unsafe_pointer(self.client.api) catch {
                const event = self.client.buffer_to_host(
                    &pjrt_buffer,
                    destination,
                ) catch |err| return map_error(err);
                return wrap_event(event);
            };
            const source: [*]const u8 = @ptrFromInt(source_address);
            if (source != destination.ptr) @memcpy(destination, source[0..destination.len]);
            return null;
        }

        const event = self.client.buffer_to_host(
            &pjrt_buffer,
            destination,
        ) catch |err| return map_error(err);
        return wrap_event(event);
    }

    fn invoke(
        interface: *Executor,
        program_handle: *anyopaque,
        inputs: []const Executor.Buffer,
        outputs: []Executor.Buffer,
        options: Executor.InvokeOptions,
    ) Executor.Error!?Executor.Event {
        const self = promote(interface);
        const non_donated_count = try validate_donation(
            inputs.len,
            options.donated_input_indices,
        );

        const non_donated = try self.client.allocator.alloc(
            i64,
            non_donated_count,
        );
        defer self.client.allocator.free(non_donated);

        var non_donated_index: usize = 0;
        for (0..inputs.len) |input_index| {
            if (contains(options.donated_input_indices, input_index)) continue;
            non_donated[non_donated_index] = @intCast(input_index);
            non_donated_index += 1;
        }

        comptime {
            if (@sizeOf(Executor.Buffer) != @sizeOf(client_mod.RawBuffer) or
                @alignOf(Executor.Buffer) != @alignOf(client_mod.RawBuffer))
            {
                @compileError("PJRT and Executor buffer handles require matching layouts");
            }
        }

        const pjrt_inputs = @as(
            [*]const client_mod.RawBuffer,
            @ptrCast(inputs.ptr),
        )[0..inputs.len];
        const pjrt_outputs = @as(
            [*]client_mod.RawBuffer,
            @ptrCast(outputs.ptr),
        )[0..outputs.len];
        const event = self.client.execute_into(
            loaded_handle(program_handle),
            pjrt_inputs,
            pjrt_outputs,
            non_donated,
            self.dispatch,
        ) catch |err| return map_error(err);
        return if (event) |completion| wrap_event(completion) else null;
    }

    fn await_event(
        interface: *Executor,
        event: Executor.Event,
    ) Executor.Error!void {
        const self = promote(interface);
        var pjrt_event = unwrap_event(event);
        self.client.await_event(&pjrt_event) catch |err| return map_error(err);
    }

    fn release_buffer(interface: *Executor, buffer: Executor.Buffer) void {
        const self = promote(interface);
        var pjrt_buffer = unwrap_buffer(buffer);
        self.client.deinit_buffer(&pjrt_buffer);
    }

    fn release_event(interface: *Executor, event: Executor.Event) void {
        const self = promote(interface);
        var pjrt_event = unwrap_event(event);
        self.client.deinit_event(&pjrt_event);
    }

    fn release_program(
        interface: *Executor,
        program_handle: *anyopaque,
    ) void {
        const self = promote(interface);
        const loaded_executable = loaded_handle(program_handle);
        self.client.deinit_executable(loaded_executable);
        self.client.allocator.destroy(loaded_executable);
    }
};

fn loaded_handle(program_handle: *anyopaque) *client_mod.LoadedExecutable {
    return @ptrCast(@alignCast(program_handle));
}

fn validate_donation(
    input_count: usize,
    donated_input_indices: []const usize,
) error{InvalidArgument}!usize {
    for (donated_input_indices, 0..) |input_index, position| {
        if (input_index >= input_count) return error.InvalidArgument;
        if (contains(donated_input_indices[0..position], input_index)) {
            return error.InvalidArgument;
        }
    }
    return input_count - donated_input_indices.len;
}

fn contains(indices: []const usize, target: usize) bool {
    for (indices) |index| {
        if (index == target) return true;
    }
    return false;
}

fn wrap_buffer(buffer: client_mod.Buffer) Executor.Buffer {
    return .{ .handle = @ptrCast(buffer.pjrt_buffer) };
}

fn unwrap_buffer(buffer: Executor.Buffer) client_mod.Buffer {
    return .{ .pjrt_buffer = @ptrCast(@alignCast(buffer.handle)) };
}

fn wrap_event(event: client_mod.Event) Executor.Event {
    return .{ .handle = @ptrCast(event.pjrt_event) };
}

fn unwrap_event(event: Executor.Event) client_mod.Event {
    return .{ .pjrt_event = @ptrCast(@alignCast(event.handle)) };
}

fn map_error(err: IntegrationError) Executor.Error {
    log.err("PJRT execution failed: {s}", .{@errorName(err)});
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidArgument,
        error.OutputArityMismatch,
        => error.InvalidArgument,
        error.Unimplemented,
        error.FunctionNotAvailable,
        error.TypedFfiUnavailable,
        => error.Unsupported,
        error.ResourceExhausted => error.ResourceExhausted,
        error.Unavailable => error.Unavailable,
        else => error.IntegrationFailure,
    };
}

test "validate_donation accepts sparse donated inputs" {
    try std.testing.expectEqual(
        @as(usize, 2),
        try validate_donation(4, &.{ 0, 2 }),
    );
}

test "validate_donation rejects repeated and out-of-range inputs" {
    try std.testing.expectError(
        error.InvalidArgument,
        validate_donation(2, &.{ 0, 0 }),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        validate_donation(2, &.{2}),
    );
}
