//! IREE implementation of the framework execution contract.

const std = @import("std");

const device_mod = @import("device");
const DType = @import("dtype").DType;
const Executor = @import("../execution.zig");
const config = @import("config.zig");
const runtime = @import("runtime.zig");

const log = std.log.scoped(.@"zg/iree_execution");

const IntegrationError = runtime.Error;

/// Synchronous execution through one IREE runtime.
pub const Execution = struct {
    interface: Executor,
    runtime: *runtime.Runtime,
    allocator: std.mem.Allocator,

    /// Configure framework execution through `runtime_instance`.
    ///
    /// Borrowed strings in `runtime_config` must outlive the execution value.
    pub fn init(
        allocator: std.mem.Allocator,
        runtime_instance: *runtime.Runtime,
        runtime_config: config.RuntimeConfig,
    ) Execution {
        return .{
            .interface = .{
                .device = .{ .platform = platform_from_driver(runtime_config.driver) },
                .vtable = &vtable,
            },
            .runtime = runtime_instance,
            .allocator = allocator,
        };
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
        dtype: DType,
        shape: []const i64,
    ) Executor.Error!Executor.Buffer {
        const self = promote(interface);
        const buffer = self.runtime.create_buffer(
            data,
            element_type(dtype),
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
        const iree_buffer = unwrap_buffer(buffer);
        self.runtime.read_buffer(&iree_buffer, destination) catch |err| return map_error(err);
        return null;
    }

    fn invoke(
        interface: *Executor,
        program_handle: *anyopaque,
        inputs: []const Executor.Buffer,
        outputs: []Executor.Buffer,
        _: Executor.InvokeOptions,
    ) Executor.Error!?Executor.Event {
        const self = promote(interface);

        comptime {
            if (@sizeOf(Executor.Buffer) != @sizeOf(runtime.Buffer) or
                @alignOf(Executor.Buffer) != @alignOf(runtime.Buffer))
            {
                @compileError("IREE and Executor buffer handles require matching layouts");
            }
        }

        const iree_inputs = @as(
            [*]const runtime.Buffer,
            @ptrCast(inputs.ptr),
        )[0..inputs.len];
        var iree_executable = unwrap_program(program_handle);
        var invocation = iree_executable.invoke(
            self.allocator,
            iree_inputs,
        ) catch |err| return map_error(err);

        if (invocation.outputs.len != outputs.len) {
            invocation.deinit();
            return error.InvalidArgument;
        }

        for (invocation.outputs, outputs) |output, *slot| {
            slot.* = wrap_buffer(output);
        }
        invocation.allocator.free(invocation.outputs);
        invocation = undefined;
        return null;
    }

    fn await_event(_: *Executor, _: Executor.Event) Executor.Error!void {
        return error.InvalidArgument;
    }

    fn release_buffer(_: *Executor, buffer: Executor.Buffer) void {
        var iree_buffer = unwrap_buffer(buffer);
        iree_buffer.deinit();
    }

    fn release_event(_: *Executor, _: Executor.Event) void {}

    fn release_program(_: *Executor, program_handle: *anyopaque) void {
        var iree_executable = unwrap_program(program_handle);
        iree_executable.deinit();
    }
};

const driver_platforms = std.StaticStringMap(device_mod.Platform).initComptime(.{
    .{ "local-sync", device_mod.Platform.cpu },
    .{ "local-task", device_mod.Platform.cpu },
    .{ "cuda", device_mod.Platform.cuda },
    .{ "hip", device_mod.Platform.rocm },
});

fn platform_from_driver(driver: []const u8) device_mod.Platform {
    return driver_platforms.get(driver) orelse .{ .name = driver };
}

fn element_type(dtype: DType) runtime.ElementType {
    return switch (dtype) {
        .bool => .bool,
        .i8 => .i8,
        .u8 => .u8,
        .i32 => .i32,
        .u32 => .u32,
        .i64 => .i64,
        .u64 => .u64,
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
    };
}

fn wrap_buffer(buffer: runtime.Buffer) Executor.Buffer {
    return .{ .handle = buffer._handle };
}

fn unwrap_buffer(buffer: Executor.Buffer) runtime.Buffer {
    return .{ ._handle = buffer.handle };
}

fn unwrap_program(program_handle: *anyopaque) runtime.Executable {
    return .{ ._state = program_handle };
}

pub fn map_error(err: IntegrationError) Executor.Error {
    log.err("IREE execution failed: {s}", .{@errorName(err)});
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.InvalidDimension => error.InvalidArgument,
        else => error.IntegrationFailure,
    };
}

test "element_type covers every PR data type" {
    inline for (std.meta.tags(DType)) |dtype| {
        _ = element_type(dtype);
    }
}

test platform_from_driver {
    try std.testing.expect(platform_from_driver("local-task").eql(.cpu));
    try std.testing.expect(platform_from_driver("cuda").eql(.cuda));
    try std.testing.expectEqualStrings(
        "vulkan",
        platform_from_driver("vulkan").name,
    );
}
