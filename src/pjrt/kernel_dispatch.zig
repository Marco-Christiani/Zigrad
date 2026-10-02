//! XLA FFI realization of store-backed kernel dispatch.

const std = @import("std");

const device = @import("device");
const kernel = @import("kernel");
const stablehlo = @import("../stablehlo.zig");
const pjrt_api = @import("../c/pjrt/api.zig");
const c = @import("../c/pjrt/c.zig").c;

const log = std.log.scoped(.@"zg/pjrt_kernel_dispatch");

pub const Options = struct {
    store: ?*const kernel.KernelStore = null,
    dispatch_registry: ?*const kernel.DispatchRegistry = null,
};

var target_registered: bool = false;

const DispatchTypeReg = struct {
    id: i64 = 0,
    registered: bool = false,
};

const dispatch_store_type_name = "zigrad.kernel.store.v1";
const dispatch_dispatch_registry_type_name = "zigrad.kernel.dispatch_registry.v1";
const dispatch_platform_type_name = "zigrad.kernel.platform.v1";

var dispatch_store_type: DispatchTypeReg = .{};
var dispatch_dispatch_registry_type: DispatchTypeReg = .{};
var dispatch_platform_type: DispatchTypeReg = .{};

pub fn register_target(api: *pjrt_api.Api, platform: device.Platform) !void {
    if (target_registered) return;
    var registered_any = false;

    if (api.ffi_extension()) |ffi_ext| {
        const any_platform = try register_dispatch_handler_for_platform(api, ffi_ext, null);
        registered_any = registered_any or any_platform;
        registered_any = try register_ffi_for_platform(api, ffi_ext, platform) or registered_any;
    }

    if (api.gpu_custom_call_extension()) |gpu_ext| {
        const gpu_registered = try register_dispatch_handler_via_gpu_extension(api, gpu_ext);
        registered_any = registered_any or gpu_registered;
    }

    if (!registered_any) return error.TypedFfiRegistrationFailed;
    target_registered = true;
}

/// Register the dispatch handler for the reported PJRT platform.
fn register_ffi_for_platform(api: *pjrt_api.Api, ffi_ext: *c.PJRT_FFI, platform: device.Platform) !bool {
    var registered_any = false;
    registered_any = try register_dispatch_handler_for_platform(
        api,
        ffi_ext,
        platform.name,
    );

    const aliases: []const []const u8 = if (platform.eql(.cpu))
        &.{ "Host", "host", "CPU", "cpu" }
    else if (platform.eql(.cuda))
        &.{ "CUDA", "cuda" }
    else
        &.{};
    for (aliases) |name| {
        registered_any = try register_dispatch_handler_for_platform(
            api,
            ffi_ext,
            name,
        ) or registered_any;
    }
    return registered_any;
}

fn register_dispatch_handler_via_gpu_extension(api: *pjrt_api.Api, gpu_ext: *c.PJRT_Gpu_Custom_Call) !bool {
    const register = gpu_ext.custom_call orelse return false;

    var args: c.PJRT_Gpu_Register_Custom_Call_Args = std.mem.zeroes(c.PJRT_Gpu_Register_Custom_Call_Args);
    args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_Gpu_Register_Custom_Call_Args);
    args.function_name = kernel.dispatch_target_name.ptr;
    args.function_name_size = kernel.dispatch_target_name.len;
    args.api_version = 1;
    args.handler_instantiate = null;
    args.handler_prepare = null;
    args.handler_initialize = null;
    args.handler_execute = @ptrCast(@constCast(&kernel_dispatch_handler));

    const pjrt_err = register(&args);
    if (pjrt_err == null) {
        log.info("registered dispatcher via gpu custom-call extension target='{s}'", .{kernel.dispatch_target_name});
        return true;
    }

    var err = pjrt_api.PjrtError.from_handle(api, pjrt_err.?);
    const msg = err.get_message(std.heap.page_allocator) catch "<failed to read message>";
    defer if (msg.ptr != "<failed to read message>".ptr) std.heap.page_allocator.free(msg);
    const code = err.get_code() catch {
        err.deinit();
        return false;
    };
    err.deinit();

    log.debug(
        "gpu custom-call register failed target='{s}' code={d} msg={s}",
        .{ kernel.dispatch_target_name, code, msg },
    );

    return code == 6;
}

fn register_dispatch_handler_for_platform(api: *pjrt_api.Api, ffi_ext: *c.PJRT_FFI, platform_name: ?[]const u8) !bool {
    var args: c.PJRT_FFI_Register_Handler_Args = std.mem.zeroes(c.PJRT_FFI_Register_Handler_Args);
    args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_FFI_Register_Handler_Args);
    args.target_name = kernel.dispatch_target_name.ptr;
    args.target_name_size = kernel.dispatch_target_name.len;
    args.handler = @ptrCast(@constCast(&kernel_dispatch_handler));
    if (platform_name) |name| {
        args.platform_name = name.ptr;
        args.platform_name_size = name.len;
    } else {
        args.platform_name = null;
        args.platform_name_size = 0;
    }
    args.traits = 0;

    const register = ffi_ext.register_handler orelse return false;
    const pjrt_err = register(&args);
    if (pjrt_err == null) {
        log.info("registered typed-ffi dispatcher target='{s}' platform='{s}'", .{ kernel.dispatch_target_name, platform_name orelse "<any>" });
        return true;
    }

    var err = pjrt_api.PjrtError.from_handle(api, pjrt_err.?);
    const msg = err.get_message(std.heap.page_allocator) catch "<failed to read message>";
    defer if (msg.ptr != "<failed to read message>".ptr) std.heap.page_allocator.free(msg);
    const code = err.get_code() catch {
        err.deinit();
        return false;
    };
    err.deinit();

    log.warn(
        "typed-ffi register_dispatch_handler_for_platform failed target='{s}' platform='{s}' code={d} msg={s}",
        .{ kernel.dispatch_target_name, platform_name orelse "<any>", code, msg },
    );

    if (code == 6) return true;
    return false;
}

fn pjrt_error_to_zig(api: *pjrt_api.Api, pjrt_err: *c.PJRT_Error) !void {
    const err = pjrt_api.PjrtError.from_handle(api, pjrt_err);
    try err.to_zig_error();
}

fn ensure_dispatch_type_id(
    api: *pjrt_api.Api,
    ffi_ext: *c.PJRT_FFI,
    type_name: []const u8,
    reg: *DispatchTypeReg,
) !i64 {
    if (reg.registered) return reg.id;

    const type_register = ffi_ext.type_register orelse return error.TypedFfiUnavailable;
    var type_info: c.PJRT_FFI_Type_Info = .{
        .deleter = null,
        .serialize = null,
        .deserialize = null,
    };

    var args: c.PJRT_FFI_Type_Register_Args = std.mem.zeroes(c.PJRT_FFI_Type_Register_Args);
    args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_FFI_Type_Register_Args);
    args.extension_start = null;
    args.type_name = type_name.ptr;
    args.type_name_size = type_name.len;
    args.type_id = 0;
    args.type_info = &type_info;

    if (type_register(&args)) |pjrt_err| {
        try pjrt_error_to_zig(api, pjrt_err);
    }

    reg.id = args.type_id;
    reg.registered = true;
    return reg.id;
}

fn add_dispatch_user_data(
    api: *pjrt_api.Api,
    ffi_ext: *c.PJRT_FFI,
    context: *c.PJRT_ExecuteContext,
    type_id: i64,
    data_ptr: *const anyopaque,
) !void {
    const add_user_data = ffi_ext.user_data_add orelse return error.TypedFfiUnavailable;

    var add_args: c.PJRT_FFI_UserData_Add_Args = std.mem.zeroes(c.PJRT_FFI_UserData_Add_Args);
    add_args.struct_size = pjrt_api.pjrt_struct_size(c.PJRT_FFI_UserData_Add_Args);
    add_args.extension_start = null;
    add_args.context = context;
    add_args.user_data = .{
        .type_id = type_id,
        // PJRT accepts a mutable user-data pointer. Dispatch only reads it.
        .data = @ptrCast(@constCast(data_ptr)),
    };

    if (add_user_data(&add_args)) |pjrt_err| {
        try pjrt_error_to_zig(api, pjrt_err);
    }
}

/// Build a PJRT execute context carrying kernel dispatch user data.
///
/// Returns `null` when no kernel dispatch is configured.
///
/// A configured context carries the store, registry, and device platform in
///  XLA FFI user-data slots.
pub fn create_execute_context(
    api: *pjrt_api.Api,
    dispatch_platform: *const device.Platform,
    options: Options,
) !?*c.PJRT_ExecuteContext {
    if (options.store == null and options.dispatch_registry == null) return null;

    const ffi_ext = api.ffi_extension() orelse return error.TypedFfiUnavailable;

    var create_args = pjrt_api.init_args(c.PJRT_ExecuteContext_Create_Args);
    create_args.context = null;
    try api.call("PJRT_ExecuteContext_Create", &create_args);

    const context = create_args.context orelse return error.PjrtReturnedNullExecuteContext;
    errdefer destroy_execute_context(api, context);

    if (options.store) |store| {
        const type_id = try ensure_dispatch_type_id(api, ffi_ext, dispatch_store_type_name, &dispatch_store_type);
        try add_dispatch_user_data(api, ffi_ext, context, type_id, @ptrCast(store));
    }

    if (options.dispatch_registry) |dreg| {
        const type_id = try ensure_dispatch_type_id(api, ffi_ext, dispatch_dispatch_registry_type_name, &dispatch_dispatch_registry_type);
        try add_dispatch_user_data(api, ffi_ext, context, type_id, @ptrCast(dreg));
    }

    // The platform pointer remains valid for every invocation using this context.
    {
        const type_id = try ensure_dispatch_type_id(api, ffi_ext, dispatch_platform_type_name, &dispatch_platform_type);
        try add_dispatch_user_data(api, ffi_ext, context, type_id, @ptrCast(dispatch_platform));
    }

    return context;
}

pub fn destroy_execute_context(api: *pjrt_api.Api, context: *c.PJRT_ExecuteContext) void {
    var destroy_args = pjrt_api.init_args(c.PJRT_ExecuteContext_Destroy_Args);
    destroy_args.context = context;
    api.call("PJRT_ExecuteContext_Destroy", &destroy_args) catch {};
}

fn lookup_dispatch_user_data_from_context(frame: *c.XLA_FFI_CallFrame, type_id_value: i64) ?*const anyopaque {
    if (type_id_value == 0) return null;

    const ffi_api = frame.api orelse return null;
    const get_context_data = ffi_api.*.XLA_FFI_ExecutionContext_Get orelse return null;

    var type_id = c.XLA_FFI_TypeId{ .type_id = type_id_value };
    var args: c.XLA_FFI_ExecutionContext_Get_Args = std.mem.zeroes(c.XLA_FFI_ExecutionContext_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_ExecutionContext_Get_Args);
    args.extension_start = null;
    args.ctx = frame.ctx;
    args.type_id = &type_id;
    args.data = null;

    if (get_context_data(&args) != null) return null;

    const data_ptr = args.data orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn lookup_store_from_context(frame: *c.XLA_FFI_CallFrame) ?*const kernel.KernelStore {
    if (!dispatch_store_type.registered) return null;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_store_type.id) orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn lookup_dispatch_registry_from_context(frame: *c.XLA_FFI_CallFrame) ?*const kernel.DispatchRegistry {
    if (!dispatch_dispatch_registry_type.registered) return null;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_dispatch_registry_type.id) orelse return null;
    return @ptrCast(@alignCast(data_ptr));
}

fn lookup_platform_from_context(frame: *c.XLA_FFI_CallFrame) ?device.Platform {
    if (!dispatch_platform_type.registered) return null;
    const data_ptr = lookup_dispatch_user_data_from_context(frame, dispatch_platform_type.id) orelse return null;
    const platform: *const device.Platform = @ptrCast(@alignCast(data_ptr));
    return platform.*;
}

/// Generic kernel dispatch handler invoked by the XLA FFI framework.
///
/// Extracts the selection key from the custom-call payload, loads the selected
///  artifact from the kernel store, builds a `DispatchContext`, and invokes the
///  provider runtime registered for that artifact.
fn kernel_dispatch_handler(frame: *c.XLA_FFI_CallFrame) callconv(.c) ?*c.XLA_FFI_Error {
    if (handle_metadata_registration_hook(frame)) return null;

    if (frame.stage != c.XLA_FFI_ExecutionStage_EXECUTE) return null;

    const maybe_kernel_key = lookup_dispatch_attr(frame.attrs, stablehlo.custom_call_payload_name);
    const kernel_key = maybe_kernel_key orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: missing payload", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    const store = lookup_store_from_context(frame) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: no kernel store in execute context", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    };

    const selection = store.get(.{ .bytes = kernel_key }) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: kernel key not found in store", c.XLA_FFI_Error_Code_NOT_FOUND);
    };

    switch (selection.candidate) {
        .provider => |stored| {
            const registry = lookup_dispatch_registry_from_context(frame);
            return dispatch_from_store(frame, stored, registry, kernel_key);
        },
        .unreplaced => {
            return make_ffi_error(frame, "zigrad kernel dispatch: store left the callable unreplaced", c.XLA_FFI_Error_Code_NOT_FOUND);
        },
    }
}

/// Resolve the selected provider runtime and dispatch its artifact.
fn dispatch_from_store(
    frame: *c.XLA_FFI_CallFrame,
    stored: kernel.ProviderCandidate,
    registry: ?*const kernel.DispatchRegistry,
    kernel_key: []const u8,
) ?*c.XLA_FFI_Error {
    const runtime = if (registry) |registered| registered.get(stored.provider_name) else null;
    if (runtime == null) {
        log.err("store dispatch: no runtime for provider '{s}'", .{stored.provider_name});
        return make_ffi_error(frame, "zigrad kernel dispatch: provider not in dispatch registry", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    }

    return execute_dispatch(
        frame,
        kernel_key,
        stored.artifact.workspace_bytes,
        stored.artifact.workspace_alignment,
        runtime.?,
        stored.artifact.data,
    );
}

/// Extract buffers, allocate workspace, and invoke a provider runtime.
fn execute_dispatch(
    frame: *c.XLA_FFI_CallFrame,
    dispatch_key: []const u8,
    workspace_bytes: usize,
    workspace_alignment: usize,
    runtime: kernel.ProviderRuntime,
    artifact_data: []const u8,
) ?*c.XLA_FFI_Error {
    var input_descs: [16]kernel.BufferDesc = undefined;
    const num_inputs = extract_buffers(frame.args.size, frame.args.types, frame.args.args, &input_descs) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: failed to extract input buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    var output_descs: [16]kernel.BufferDesc = undefined;
    const num_outputs = extract_ret_buffers(frame.rets.size, frame.rets.types, frame.rets.rets, &output_descs) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: failed to extract output buffers", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    };

    const platform = lookup_platform_from_context(frame) orelse {
        return make_ffi_error(frame, "zigrad kernel dispatch: no device platform in execute context", c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    };

    var workspace_ptr: ?*anyopaque = null;
    if (workspace_bytes > 0) {
        if (workspace_alignment == 0 or !std.math.isPowerOfTwo(workspace_alignment)) {
            log.err("invalid workspace alignment {d} for '{s}'", .{ workspace_alignment, dispatch_key });
            return make_ffi_error(frame, "zigrad kernel dispatch: invalid workspace alignment", c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
        }
        workspace_ptr = alloc_device_memory(frame, workspace_bytes, workspace_alignment);
        if (workspace_ptr == null) {
            log.err("workspace allocation failed: {d} bytes for '{s}'", .{ workspace_bytes, dispatch_key });
            return make_ffi_error(frame, "zigrad kernel dispatch: workspace allocation failed", c.XLA_FFI_Error_Code_RESOURCE_EXHAUSTED);
        }
        log.debug("allocated {d} bytes workspace for '{s}'", .{ workspace_bytes, dispatch_key });
    }
    defer if (workspace_ptr) |ptr| {
        free_device_memory(frame, ptr, workspace_bytes);
        log.debug("freed {d} bytes workspace for '{s}'", .{ workspace_bytes, dispatch_key });
    };

    const ctx = kernel.DispatchContext{
        .inputs = input_descs[0..num_inputs],
        .outputs = output_descs[0..num_outputs],
        .device = .{
            .platform = platform,
            .ordinal = get_device_ordinal(frame),
        },
        .stream = get_stream(frame),
        .workspace = workspace_ptr,
        .workspace_bytes_required = workspace_bytes,
        .allocator = std.heap.c_allocator,
    };

    runtime.dispatch(artifact_data, dispatch_key, ctx) catch |err| {
        log.err("kernel dispatch failed for '{s}': {s}", .{ dispatch_key, @errorName(err) });
        return dispatch_error_to_ffi(frame, err);
    };
    return null;
}

fn dispatch_error_to_ffi(frame: *c.XLA_FFI_CallFrame, err: kernel.DispatchError) ?*c.XLA_FFI_Error {
    return make_ffi_error(frame, "zigrad kernel dispatch failed", dispatch_error_code(err));
}

fn dispatch_error_code(err: kernel.DispatchError) c.XLA_FFI_Error_Code {
    return switch (err) {
        error.UnsupportedDType,
        error.ShapeMismatch,
        => c.XLA_FFI_Error_Code_INVALID_ARGUMENT,

        error.ProviderLoadFailed,
        error.WorkspaceUnavailable,
        error.UnsupportedDevice,
        => c.XLA_FFI_Error_Code_FAILED_PRECONDITION,

        error.DispatchFailed,
        error.OutOfMemory,
        => c.XLA_FFI_Error_Code_INTERNAL,
    };
}

/// Extract input buffers from an FFI args structure into BufferDesc array.
fn extract_buffers(
    size: i64,
    types: [*]const c.XLA_FFI_ArgType,
    args_ptr: [*]const ?*anyopaque,
    out: *[16]kernel.BufferDesc,
) ?usize {
    if (size < 0) return null;
    const count: usize = @intCast(size);
    if (count > 16) return null;
    for (0..count) |i| {
        if (types[i] != c.XLA_FFI_ArgType_BUFFER) return null;
        const ptr = args_ptr[i] orelse return null;
        const buf: *c.XLA_FFI_Buffer = @ptrCast(@alignCast(ptr));
        out[i] = ffi_buffer_to_desc(buf) orelse return null;
    }
    return count;
}

/// Extract output buffers from an FFI rets structure into BufferDesc array.
fn extract_ret_buffers(
    size: i64,
    types: [*]const c.XLA_FFI_RetType,
    rets_ptr: [*]const ?*anyopaque,
    out: *[16]kernel.BufferDesc,
) ?usize {
    if (size < 0) return null;
    const count: usize = @intCast(size);
    if (count > 16) return null;
    for (0..count) |i| {
        if (types[i] != c.XLA_FFI_RetType_BUFFER) return null;
        const ptr = rets_ptr[i] orelse return null;
        const buf: *c.XLA_FFI_Buffer = @ptrCast(@alignCast(ptr));
        out[i] = ffi_buffer_to_desc(buf) orelse return null;
    }
    return count;
}

/// Convert an XLA FFI buffer to a provider-agnostic BufferDesc.
fn ffi_buffer_to_desc(buf: *c.XLA_FFI_Buffer) ?kernel.BufferDesc {
    const dtype = ffi_dtype_to_kernel_dtype(buf.dtype) orelse return null;
    const rank: usize = @intCast(buf.rank);
    return .{
        .data = @ptrCast(buf.data orelse return null),
        .dtype = dtype,
        .dims = buf.dims[0..rank],
        .rank = rank,
    };
}

/// Map XLA FFI element types to kernel.DType.
fn ffi_dtype_to_kernel_dtype(xla_dtype: c.XLA_FFI_DataType) ?kernel.DType {
    if (xla_dtype == c.XLA_FFI_DataType_F16) return .f16;
    if (xla_dtype == c.XLA_FFI_DataType_BF16) return .bf16;
    if (xla_dtype == c.XLA_FFI_DataType_F32) return .f32;
    if (xla_dtype == c.XLA_FFI_DataType_F64) return .f64;
    if (xla_dtype == c.XLA_FFI_DataType_S8) return .i8;
    if (xla_dtype == c.XLA_FFI_DataType_U8) return .u8;
    if (xla_dtype == c.XLA_FFI_DataType_S32) return .i32;
    if (xla_dtype == c.XLA_FFI_DataType_S64) return .i64;
    if (xla_dtype == c.XLA_FFI_DataType_U32) return .u32;
    if (xla_dtype == c.XLA_FFI_DataType_U64) return .u64;
    return null;
}

/// Extract the GPU stream handle from an FFI call frame.
fn get_stream(frame: *c.XLA_FFI_CallFrame) ?*anyopaque {
    const ffi_api = frame.api orelse return null;
    const get_stream_fn = ffi_api.*.XLA_FFI_Stream_Get orelse return null;
    var args: c.XLA_FFI_Stream_Get_Args = std.mem.zeroes(c.XLA_FFI_Stream_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_Stream_Get_Args);
    args.ctx = frame.ctx;
    const ffi_err = get_stream_fn(&args);
    if (ffi_err != null) return null;
    return args.stream;
}

fn get_device_ordinal(frame: *c.XLA_FFI_CallFrame) i32 {
    const ffi_api = frame.api orelse return 0;
    const get_ordinal = ffi_api.*.XLA_FFI_DeviceOrdinal_Get orelse return 0;
    var args: c.XLA_FFI_DeviceOrdinal_Get_Args = std.mem.zeroes(c.XLA_FFI_DeviceOrdinal_Get_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_DeviceOrdinal_Get_Args);
    args.ctx = frame.ctx;
    const err = get_ordinal(&args);
    if (err != null) return 0;
    return args.device_ordinal;
}

/// Allocate device memory from XLA's BFC pool via the FFI execution context.
///
/// Returns null if the API is unavailable or allocation fails. The returned
///  pointer refers to the BFC pool. Callers pair each successful allocation
///  with `free_device_memory`.
fn alloc_device_memory(frame: *c.XLA_FFI_CallFrame, size: usize, alignment: usize) ?*anyopaque {
    const ffi_api = frame.api orelse return null;
    const alloc_fn = ffi_api.*.XLA_FFI_DeviceMemory_Allocate orelse return null;
    var args: c.XLA_FFI_DeviceMemory_Allocate_Args = std.mem.zeroes(c.XLA_FFI_DeviceMemory_Allocate_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_DeviceMemory_Allocate_Args);
    args.ctx = frame.ctx;
    args.size = size;
    args.alignment = alignment;
    const ffi_err = alloc_fn(&args);
    if (ffi_err != null) {
        log.warn("XLA_FFI_DeviceMemory_Allocate failed for {d} bytes", .{size});
        return null;
    }
    return args.data;
}

/// Free device memory allocated by `alloc_device_memory`.
///
/// Returns memory to XLA's BFC pool instead of calling `cudaFree`.
///
/// A failed release logs a warning. Workspace cleanup cannot propagate errors.
fn free_device_memory(frame: *c.XLA_FFI_CallFrame, data: *anyopaque, size: usize) void {
    const ffi_api = frame.api orelse {
        log.warn("free_device_memory: FFI API unavailable", .{});
        return;
    };
    const free_fn = ffi_api.*.XLA_FFI_DeviceMemory_Free orelse {
        log.warn("free_device_memory: XLA_FFI_DeviceMemory_Free unavailable", .{});
        return;
    };
    var args: c.XLA_FFI_DeviceMemory_Free_Args = std.mem.zeroes(c.XLA_FFI_DeviceMemory_Free_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_DeviceMemory_Free_Args);
    args.ctx = frame.ctx;
    args.size = size;
    args.data = data;
    const ffi_err = free_fn(&args);
    if (ffi_err != null) {
        log.warn("XLA_FFI_DeviceMemory_Free failed for {d} bytes", .{size});
    }
}

fn handle_metadata_registration_hook(frame: *c.XLA_FFI_CallFrame) bool {
    const ext = frame.extension_start orelse return false;
    const ext_ptr: *c.XLA_FFI_Extension_Base = @ptrCast(ext);
    if (ext_ptr.type != c.XLA_FFI_Extension_Metadata) return false;

    const metadata_ext: *c.XLA_FFI_Metadata_Extension = @fieldParentPtr("extension_base", ext_ptr);
    const metadata = metadata_ext.metadata orelse return false;
    const metadata_ptr: *c.XLA_FFI_Metadata = @ptrCast(metadata);
    metadata_ptr.api_version.major_version = c.XLA_FFI_API_MAJOR;
    metadata_ptr.api_version.minor_version = c.XLA_FFI_API_MINOR;
    return true;
}

fn lookup_dispatch_attr(attrs: c.XLA_FFI_Attrs, name: []const u8) ?[]const u8 {
    var i: usize = 0;
    while (i < @as(usize, @intCast(attrs.size))) : (i += 1) {
        const name_span = attrs.names[i][0];
        const key = name_span.ptr[0..name_span.len];
        if (!std.mem.eql(u8, key, name)) continue;

        if (attrs.types[i] != c.XLA_FFI_AttrType_STRING) return null;
        const attr_value = attrs.attrs[i] orelse return null;
        const span: *c.XLA_FFI_ByteSpan = @ptrCast(@alignCast(attr_value));
        return span.ptr[0..span.len];
    }
    return null;
}

fn make_ffi_error(frame: *c.XLA_FFI_CallFrame, message: [:0]const u8, code: c.XLA_FFI_Error_Code) ?*c.XLA_FFI_Error {
    const ffi_api = frame.api orelse return null;
    const create_error = ffi_api.*.XLA_FFI_Error_Create orelse return null;
    var args: c.XLA_FFI_Error_Create_Args = std.mem.zeroes(c.XLA_FFI_Error_Create_Args);
    args.struct_size = @sizeOf(c.XLA_FFI_Error_Create_Args);
    args.message = message.ptr;
    args.errc = code;
    return create_error(&args);
}

test "dispatch_error_to_ffi returns null when frame has no API" {
    var frame: c.XLA_FFI_CallFrame = std.mem.zeroes(c.XLA_FFI_CallFrame);
    try std.testing.expect(dispatch_error_to_ffi(&frame, error.ShapeMismatch) == null);
}

test "dispatch_error_code maps invalid-argument class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_INVALID_ARGUMENT);
    try std.testing.expectEqual(expected, dispatch_error_code(error.ShapeMismatch));
}

test "dispatch_error_code maps failed-precondition class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_FAILED_PRECONDITION);
    try std.testing.expectEqual(expected, dispatch_error_code(error.WorkspaceUnavailable));
}

test "dispatch_error_code maps internal class" {
    const expected: c.XLA_FFI_Error_Code = @intCast(c.XLA_FFI_Error_Code_INTERNAL);
    try std.testing.expectEqual(expected, dispatch_error_code(error.DispatchFailed));
}

test "lookup_dispatch_user_data_from_context returns payload" {
    const testing = std.testing;

    const FfiStub = struct {
        var expected_type_id: i64 = 0;
        var payload: ?*anyopaque = null;

        fn execution_context_get(raw_args: [*c]c.XLA_FFI_ExecutionContext_Get_Args) callconv(.c) ?*c.XLA_FFI_Error {
            const args: *c.XLA_FFI_ExecutionContext_Get_Args = @ptrCast(raw_args);
            if (args.type_id != null and args.type_id.*.type_id == expected_type_id) {
                args.data = payload;
            } else {
                args.data = null;
            }
            return null;
        }
    };

    var payload_byte: u8 = 42;
    FfiStub.expected_type_id = 123;
    FfiStub.payload = @ptrCast(&payload_byte);

    var api: c.XLA_FFI_Api = std.mem.zeroes(c.XLA_FFI_Api);
    api.struct_size = @sizeOf(c.XLA_FFI_Api);
    api.XLA_FFI_ExecutionContext_Get = FfiStub.execution_context_get;

    var frame: c.XLA_FFI_CallFrame = std.mem.zeroes(c.XLA_FFI_CallFrame);
    frame.struct_size = @sizeOf(c.XLA_FFI_CallFrame);
    frame.api = &api;
    frame.ctx = @ptrFromInt(1);

    const looked_up = lookup_dispatch_user_data_from_context(&frame, FfiStub.expected_type_id);
    try testing.expect(looked_up != null);
    try testing.expectEqual(@intFromPtr(&payload_byte), @intFromPtr(looked_up.?));
}

test "lookup_dispatch_user_data_from_context returns null without api" {
    const testing = std.testing;

    var frame: c.XLA_FFI_CallFrame = std.mem.zeroes(c.XLA_FFI_CallFrame);
    frame.struct_size = @sizeOf(c.XLA_FFI_CallFrame);
    frame.api = null;

    try testing.expect(lookup_dispatch_user_data_from_context(&frame, 1) == null);
}
