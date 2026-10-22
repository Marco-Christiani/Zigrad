//! Lifecycle for the optional IREE runtime.
//!
//! Raw IREE types remain under `src/c/iree`. This module exposes synchronous
//!  execution through type-erased handles.

const std = @import("std");
const abi = @import("iree_abi");
const build_options = @import("build_options");
const RuntimeConfig = @import("config.zig").RuntimeConfig;

const log = std.log.scoped(.@"zg/iree_runtime");

/// Element types accepted by the current IREE buffer API.
pub const ElementType = abi.ElementType;

/// Failures exposed by the IREE runtime integration.
pub const Error = abi.Error;

/// Mechanism used to construct an IREE HAL device.
pub const DeviceConstruction = enum {
    registered,
    embedded_elf_sync,
};

/// Alignment in bytes of owned VMFB storage. Allocators receive it with every
///  VMFB allocation and release.
pub const bytecode_alignment_bytes = 16;

/// VMFB storage retained for the lifetime of a loaded executable.
pub const Bytecode = union(enum) {
    /// Storage borrowed from memory that outlives the executable.
    borrowed: []const u8,

    /// Storage released with its allocator when the executable is released.
    ///
    /// `bytes` was allocated with `bytecode_alignment_bytes`, which the slice
    ///  type carries into the release.
    owned: struct {
        bytes: []align(bytecode_alignment_bytes) u8,
        allocator: std.mem.Allocator,
    },

    fn bytes(self: Bytecode) []const u8 {
        return switch (self) {
            .borrowed => |storage| storage,
            .owned => |storage| storage.bytes,
        };
    }

    /// Release owned storage. Borrowed storage is unchanged.
    pub fn deinit(self: *Bytecode) void {
        switch (self.*) {
            .borrowed => {},
            .owned => |storage| storage.allocator.free(storage.bytes),
        }
        self.* = undefined;
    }
};

/// IREE buffer view released with `deinit`.
pub const Buffer = struct {
    /// Type-erased integration handle.
    _handle: *anyopaque,

    /// Release this buffer view.
    pub fn deinit(self: *Buffer) void {
        abi.buffer_view_release(buffer_view(self));
        self.* = undefined;
    }

    /// Return the number of logical elements in this buffer.
    pub fn element_count(self: *const Buffer) usize {
        return abi.buffer_view_element_count(buffer_view(self));
    }

    /// Return this buffer's element type when Zigrad supports it.
    pub fn element_type(self: *const Buffer) Error!ElementType {
        return try abi.buffer_view_element_type(buffer_view(self));
    }
};

/// Results from one synchronous invocation.
pub const Invocation = struct {
    /// Output buffers in function-result order.
    outputs: []Buffer,

    /// Allocator used to free the output slice.
    allocator: std.mem.Allocator,

    /// Release every output buffer and the containing slice.
    pub fn deinit(self: *Invocation) void {
        for (self.outputs) |*output| output.deinit();
        self.allocator.free(self.outputs);
        self.* = undefined;
    }
};

const ExecutableState = struct {
    session: *abi.SessionHandle,
    function: *abi.FunctionHandle,
    bytecode: Bytecode,
    allocator: std.mem.Allocator,
};

/// One loaded VMFB and its resolved entry function.
///
/// `deinit` releases the VMFB bytes. Call it before releasing the `Runtime`
///  that loaded this executable.
pub const Executable = struct {
    /// Type-erased IREE state.
    _state: *anyopaque,

    /// Release the session and VMFB bytes.
    pub fn deinit(self: *Executable) void {
        const state = executable_state(self);
        abi.function_release(state.function);
        abi.session_release(state.session);
        state.bytecode.deinit();
        state.allocator.destroy(state);
        self.* = undefined;
    }

    /// Invoke this executable synchronously with borrowed input buffers.
    pub fn invoke(
        self: *Executable,
        allocator: std.mem.Allocator,
        inputs: []const Buffer,
    ) Error!Invocation {
        const state = executable_state(self);
        const call = try abi.call_init(allocator, state.session, state.function);
        defer abi.call_deinit(call);

        for (inputs) |*input| {
            try abi.call_push_buffer_view_input(call, buffer_view(input));
        }
        try abi.call_invoke(call);

        var outputs: std.ArrayList(Buffer) = .empty;
        defer outputs.deinit(allocator);
        errdefer for (outputs.items) |*output| output.deinit();

        while (try abi.call_pop_buffer_view_output(call)) |view| {
            errdefer abi.buffer_view_release(view);
            try outputs.append(allocator, .{ ._handle = view });
        }

        return .{
            .outputs = try outputs.toOwnedSlice(allocator),
            .allocator = allocator,
        };
    }
};

const RuntimeState = struct {
    instance: *abi.InstanceHandle,
    device: *abi.DeviceHandle,
    allocator: std.mem.Allocator,
};

/// One configured IREE runtime instance and its synchronous HAL device.
pub const Runtime = struct {
    /// Type-erased IREE state.
    _state: *anyopaque,

    /// Create a runtime and one device from explicit configuration.
    pub fn init(
        allocator: std.mem.Allocator,
        comptime device_construction: DeviceConstruction,
        config: RuntimeConfig,
    ) Error!Runtime {
        if (device_construction == .embedded_elf_sync and !build_options.has_iree_embedded_elf)
            @compileError("embedded ELF device construction requires -Diree-embedded-elf=true");

        const instance = switch (device_construction) {
            .registered => try abi.instance_create(),
            .embedded_elf_sync => try abi.instance_create_without_drivers(),
        };
        errdefer abi.instance_release(instance);

        const device = switch (device_construction) {
            .registered => try abi.create_default_device(instance, config.driver),
            .embedded_elf_sync => try abi.create_embedded_elf_sync_device(),
        };
        errdefer abi.device_release(device);

        const state = try allocator.create(RuntimeState);
        state.* = .{
            .instance = instance,
            .device = device,
            .allocator = allocator,
        };

        switch (device_construction) {
            .registered => log.info("initialized IREE driver '{s}'", .{config.driver}),
            .embedded_elf_sync => log.info("initialized IREE embedded ELF local-sync device", .{}),
        }
        return .{ ._state = state };
    }

    /// Release this runtime's device and instance.
    ///
    /// All buffers and executables created by this runtime must already be
    ///  released.
    pub fn deinit(self: *Runtime) void {
        const state = runtime_state(self);
        abi.device_release(state.device);
        abi.instance_release(state.instance);
        state.allocator.destroy(state);
        self.* = undefined;
    }

    /// Load VMFB storage and resolve one fully qualified entry function.
    ///
    /// On success, `Executable.deinit` releases owned storage. The caller
    ///  remains responsible for `bytecode` when loading fails.
    pub fn load(
        self: *Runtime,
        allocator: std.mem.Allocator,
        bytecode: *Bytecode,
        entry_name: []const u8,
    ) Error!Executable {
        const runtime = runtime_state(self);
        switch (bytecode.*) {
            .owned => |storage| std.debug.assert(@intFromPtr(storage.bytes.ptr) % bytecode_alignment_bytes == 0),
            .borrowed => {},
        }
        const session = try abi.session_create(runtime.instance, runtime.device);
        errdefer abi.session_release(session);

        try abi.session_append_module(session, bytecode.bytes());
        const function = try abi.session_lookup_function(
            allocator,
            session,
            entry_name,
        );
        errdefer abi.function_release(function);

        const state = try allocator.create(ExecutableState);
        state.* = .{
            .session = session,
            .function = function,
            .bytecode = bytecode.*,
            .allocator = allocator,
        };
        bytecode.* = undefined;
        return .{ ._state = state };
    }

    /// Create a device buffer by copying row-major host data.
    pub fn create_buffer(
        self: *Runtime,
        data: []const u8,
        element_type: ElementType,
        shape: []const i64,
    ) Error!Buffer {
        const view = try abi.buffer_view_create_from_host(
            runtime_state(self).device,
            data,
            element_type,
            shape,
        );
        return .{ ._handle = view };
    }

    /// Copy one device buffer's contents into host memory.
    pub fn read_buffer(
        self: *Runtime,
        buffer: *const Buffer,
        destination: []u8,
    ) Error!void {
        try abi.buffer_view_to_host(
            runtime_state(self).device,
            buffer_view(buffer),
            destination,
        );
    }
};

fn buffer_view(buffer: *const Buffer) *abi.BufferHandle {
    return @ptrCast(@alignCast(buffer._handle));
}

fn executable_state(executable: *const Executable) *ExecutableState {
    return @ptrCast(@alignCast(executable._state));
}

fn runtime_state(runtime: *const Runtime) *RuntimeState {
    return @ptrCast(@alignCast(runtime._state));
}

test "ElementType reports storage widths" {
    try std.testing.expectEqual(@as(usize, 1), ElementType.i8.byte_width());
    try std.testing.expectEqual(@as(usize, 2), ElementType.f16.byte_width());
    try std.testing.expectEqual(@as(usize, 4), ElementType.f32.byte_width());
    try std.testing.expectEqual(@as(usize, 8), ElementType.i64.byte_width());
}

test "IREE API does not export external declarations" {
    try std.testing.expect(!@hasDecl(abi, "Allocator"));
    try std.testing.expect(!@hasDecl(abi, "Status"));
    try std.testing.expect(!@hasDecl(abi, "VmFunction"));
    try std.testing.expect(!@hasDecl(abi, "RuntimeCall"));
    try std.testing.expect(!@hasDecl(abi, "HalDevice"));
    try std.testing.expect(!@hasDecl(abi, "HalBufferView"));
    try std.testing.expect(!@hasDecl(abi, "HalElementType"));
    try std.testing.expect(!@hasDecl(abi, "HalDim"));
}
