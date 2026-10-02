//! PJRT plugin, client, compilation, and integration-specific capabilities.
//!
//! `Client` loads one plugin and releases its PJRT objects through explicit
//!  initialization and deinitialization.

// TODO(pjrt): Verify the async contract before exposing async manager APIs.
const std = @import("std");

const device_mod = @import("device");
const DType = @import("dtype").DType;
const stablehlo = @import("../stablehlo.zig");
const dispatch = @import("kernel_dispatch.zig");
pub const config = @import("config.zig");
const plugin = @import("../c/pjrt/plugin.zig");
const pjrt_api = @import("../c/pjrt/api.zig");
const pjrt_types = @import("../c/pjrt/types.zig");
const log = std.log.scoped(.@"zg/pjrt_client");

pub const LoadedExecutable = pjrt_types.LoadedExecutable;
pub const Buffer = pjrt_types.Buffer;
pub const RawBuffer = pjrt_types.RawBuffer;
pub const Device = pjrt_types.Device;
pub const Event = pjrt_types.Event;
pub const ExecuteResult = pjrt_types.ExecuteResult;

pub const CompileOptions = struct {
    num_replicas: u32 = 1,
    num_partitions: u32 = 1,
};

pub const DispatchOptions = dispatch.Options;

pub const InitOptions = config.InitOptions;

/// Loaded PJRT plugin and its client.
pub const Client = struct {
    api: *pjrt_api.Api,
    client: pjrt_types.Client,
    allocator: std.mem.Allocator,
    platform: device_mod.Platform,
    kernel_dispatch_registered: bool,
    plugin_options: config.PluginOptions,

    /// Load a PJRT plugin and create its client.
    ///
    /// `plugin_path` points to a PJRT plugin DSO. The caller resolves process
    ///  configuration into `options`.
    ///
    /// Initialization does not inspect the process environment.
    pub fn init(
        allocator: std.mem.Allocator,
        plugin_path: []const u8,
        options: InitOptions,
    ) !Client {
        const api_ptr = try allocator.create(pjrt_api.Api);
        errdefer allocator.destroy(api_ptr);

        api_ptr.* = try plugin.load_plugin(plugin_path, options.plugin);
        errdefer plugin.unload_plugin(api_ptr.*, options.plugin);

        var client = switch (options.client) {
            .default => try pjrt_types.Client.create(api_ptr),
            .xla_cpu => |cpu| try pjrt_types.Client.create_cpu_with_device_count(
                api_ptr,
                cpu.device_count,
            ),
            .xla_gpu => |gpu| try pjrt_types.Client.create_xla_gpu_with_memory_fraction(
                api_ptr,
                gpu.memory_fraction,
            ),
        };
        errdefer client.deinit();

        const platform = device_mod.Platform{
            .name = try client.get_platform_name(),
        };

        const self = Client{
            .api = api_ptr,
            .client = client,
            .allocator = allocator,
            .platform = platform,
            .kernel_dispatch_registered = false,
            .plugin_options = options.plugin,
        };
        log.info("plugin loaded: {s}", .{plugin_path});
        log.info("platform: {s}, typed-ffi: {s}", .{
            self.platform.name,
            if (self.api.ffi_extension() != null) "available" else "unavailable",
        });
        return self;
    }

    pub fn deinit(self: *Client) void {
        self.client.deinit();
        plugin.unload_plugin(self.api.*, self.plugin_options);
        self.allocator.destroy(self.api);
    }

    /// Return every device reported by this client.
    pub fn get_devices(self: *Client, allocator: std.mem.Allocator) ![]Device {
        return try self.client.get_devices(allocator);
    }

    /// Return the underlying PJRT client handle.
    pub fn get_client(self: *Client) *pjrt_types.Client {
        return &self.client;
    }

    /// Compile StableHLO bytes to a loaded executable.
    ///
    /// `encoding` selects the wire format (text vs binary).
    pub fn compile(
        self: *Client,
        device: *const Device,
        ir_bytes: []const u8,
        encoding: stablehlo.Encoding,
        options: CompileOptions,
    ) !LoadedExecutable {
        log.info("compile: {d:.1}KB {s}, platform={s}", .{
            @as(f64, @floatFromInt(ir_bytes.len)) / 1024.0,
            @tagName(encoding),
            self.platform.name,
        });

        // TODO(pjrt): Add compile timing when Client receives the I/O state required
        //  to construct a monotonic timer.
        const compile_opts_pb = try build_compile_options_proto(self.allocator, options);
        defer self.allocator.free(compile_opts_pb);

        // TODO(pjrt): Encode the selected device in compile options when the
        //  PJRT device-assignment contract is exposed here.
        _ = device;
        const executable = try self.client.compile("mlir", ir_bytes, compile_opts_pb);

        return executable;
    }

    /// Compile and serialize the resulting executable (for caching).
    pub fn compile_serialized(
        self: *Client,
        device: *const Device,
        ir_bytes: []const u8,
        encoding: stablehlo.Encoding,
        options: CompileOptions,
    ) ![]u8 {
        var exe = try self.compile(device, ir_bytes, encoding, options);
        defer exe.deinit(self.api);
        return try exe.serialize(self.api, self.allocator);
    }

    /// Load a serialized executable.
    pub fn load_serialized_executable(
        self: *Client,
        serialized_executable: []const u8,
        overridden_compile_options: ?[]const u8,
    ) !LoadedExecutable {
        return try self.client.deserialize_and_load(serialized_executable, overridden_compile_options);
    }

    /// Create a buffer on device from host data.
    pub fn buffer_from_host(
        self: *Client,
        device: *const Device,
        data: []const u8,
        dtype: DType,
        shape: []const i64,
    ) !Buffer {
        const buf_type = dtype_to_buffer_type(dtype);
        return try self.client.buffer_from_host(device, data, buf_type, shape);
    }

    pub fn execute(self: *Client, exe: *LoadedExecutable, allocator: std.mem.Allocator, inputs: []const Buffer, options: DispatchOptions) !ExecuteResult {
        const execute_context = try dispatch.create_execute_context(self.api, &self.platform, options);
        defer if (execute_context) |ctx| dispatch.destroy_execute_context(self.api, ctx);
        return try exe.execute_with_context(self.api, allocator, inputs, execute_context);
    }

    pub fn execute_into(self: *Client, exe: *LoadedExecutable, input_ptrs: []const RawBuffer, output_ptrs: []RawBuffer, non_donatable: ?[]const i64, options: DispatchOptions) !?Event {
        const execute_context = try dispatch.create_execute_context(self.api, &self.platform, options);
        defer if (execute_context) |ctx| dispatch.destroy_execute_context(self.api, ctx);
        return try exe.execute_into_opts_with_context(self.api, input_ptrs, output_ptrs, non_donatable, execute_context);
    }

    pub fn deinit_buffer(self: *Client, buf: *Buffer) void {
        buf.deinit(self.api);
    }

    pub fn deinit_event(self: *Client, ev: *Event) void {
        ev.deinit(self.api);
    }

    pub fn deinit_executable(self: *Client, exe: *LoadedExecutable) void {
        exe.deinit(self.api);
    }

    pub fn buffer_to_host(self: *Client, buf: *Buffer, dst: []u8) !Event {
        return try buf.to_host(self.api, dst);
    }

    pub fn await_event(self: *Client, ev: *Event) !void {
        return try ev.await_(self.api);
    }

    pub fn buffer_unsafe_pointer(self: *Client, buf: *const Buffer) !usize {
        return try buf.unsafe_pointer(self.api);
    }

    pub fn buffer_is_on_cpu(self: *Client, buf: *const Buffer) !bool {
        return try buf.is_on_cpu(self.api);
    }

    pub fn executable_memory_stats(self: *Client, exe: *LoadedExecutable) !LoadedExecutable.CompiledMemoryStats {
        return try exe.get_compiled_memory_stats(self.api);
    }

    pub fn has_typed_ffi(self: *Client) bool {
        return self.api.ffi_extension() != null;
    }

    pub fn device_kind(self: *Client, device: *const Device) ![]const u8 {
        return try device.get_kind(self.api);
    }

    pub fn is_cuda(self: *Client) bool {
        return self.platform.eql(.cuda);
    }

    /// Require the XLA FFI extension used by kernelized custom calls.
    pub fn require_typed_ffi(self: *Client) !void {
        if (!self.has_typed_ffi()) return error.TypedFfiUnavailable;
    }

    /// Register the Zigrad kernel dispatch target.
    ///
    /// Dispatch entries are resolved at execution time via `DispatchOptions`.
    pub fn register_kernel_dispatcher(self: *Client) !void {
        try self.require_typed_ffi();
        if (self.kernel_dispatch_registered) return;

        try dispatch.register_target(self.api, self.platform);

        self.kernel_dispatch_registered = true;
    }

    pub fn device_memory_stats(self: *Client, device: *const Device) !Device.MemoryStats {
        return try device.get_memory_stats(self.api);
    }
};

fn write_varint(writer: anytype, value: u64) !void {
    var v = value;
    while (true) {
        const byte: u8 = @intCast(v & 0x7F);
        v >>= 7;
        if (v == 0) {
            try writer.writeByte(byte);
            return;
        }
        try writer.writeByte(byte | 0x80);
    }
}

fn dtype_to_buffer_type(dtype: DType) pjrt_types.BufferType {
    return switch (dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .i8 => .i8,
        .u8 => .u8,
        .i32 => .i32,
        .i64 => .i64,
        .u32 => .u32,
        .u64 => .u64,
        .bool => .i32,
    };
}

/// Build a minimal CompileOptionsProto for PJRT (protobuf wire format).
fn build_compile_options_proto(allocator: std.mem.Allocator, options: CompileOptions) ![]u8 {
    var build_aw: std.Io.Writer.Allocating = .init(allocator);
    defer build_aw.deinit();
    const b = &build_aw.writer;

    // ExecutableBuildOptionsProto:
    //   int64 num_replicas = 4;
    //   int64 num_partitions = 5;
    try b.writeByte((4 << 3) | 0);
    try write_varint(b, options.num_replicas);
    try b.writeByte((5 << 3) | 0);
    try write_varint(b, options.num_partitions);

    const build_bytes = build_aw.writer.buffered();

    var out_aw: std.Io.Writer.Allocating = .init(allocator);
    errdefer out_aw.deinit();
    const w = &out_aw.writer;

    // CompileOptionsProto:
    //   ExecutableBuildOptionsProto executable_build_options = 3;
    try w.writeByte((3 << 3) | 2);
    try write_varint(w, build_bytes.len);
    try w.writeAll(build_bytes);

    return try out_aw.toOwnedSlice();
}
