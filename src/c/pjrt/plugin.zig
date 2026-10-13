//! PJRT Plugin Loader
//!
//! Loads PJRT plugins from explicit paths using dlopen.
//!
//! Usage:
//!   const api = try load_plugin("/path/to/pjrt_cpu_plugin.so", .{});
//!   defer unload_plugin(api, .{});
const std = @import("std");
const api_mod = @import("api.zig");
const Api = api_mod.Api;
const c = @import("c.zig").c;
const config = @import("../../pjrt/config.zig");

var cached_api: ?Api = null;
var cached_path: ?[]const u8 = null;
var cached_refcount: usize = 0;

fn dl_err_msg() []const u8 {
    const p = c.dlerror() orelse return "dlerror() returned null";
    return std.mem.span(p);
}

fn log_dladdr(label: []const u8, addr: *const anyopaque, debug: bool) void {
    if (!debug) return;
    var info: c.Dl_info = undefined;
    if (c.dladdr(addr, &info) == 0) {
        std.debug.print("[pjrt-debug] dladdr {s}: <unresolved> addr={*}\n", .{ label, addr });
        return;
    }
    const fname = if (info.dli_fname) |p| std.mem.span(p) else "<null>";
    const sname = if (info.dli_sname) |p| std.mem.span(p) else "<null>";
    std.debug.print("[pjrt-debug] dladdr {s}: dso={s} sym={s} addr={*}\n", .{ label, fname, sname, addr });
}

fn canonicalize_path(path: []const u8) ![]const u8 {
    // Keep caller-provided path semantics (including symlinks) so plugin
    // RUNPATH relative lookups resolve against the assembled runtime inputs.
    return try std.heap.page_allocator.dupe(u8, path);
}

/// Load a PJRT plugin from an explicit path.
///
/// The caller owns path discovery and loader policy. This function does not
///  inspect the process environment.
pub fn load_plugin(path: []const u8, options: config.PluginOptions) !Api {
    const debug = options.debug;

    const canonical = try canonicalize_path(path);
    errdefer std.heap.page_allocator.free(canonical);

    // CUDA plugins resolve driver symbols from host-provided libraries.
    const base = std.fs.path.basename(canonical);
    const wants_cuda = std.mem.indexOf(u8, base, "gpu") != null or std.mem.indexOf(u8, base, "cuda") != null;
    if (wants_cuda) {
        _ = try preload_host_nvidia(debug);
    }

    if (cached_path) |existing| {
        if (!std.mem.eql(u8, existing, canonical)) {
            if (debug) {
                std.debug.print("[pjrt-debug] load_plugin called with different path\n", .{});
                std.debug.print("[pjrt-debug] existing={s}\n", .{existing});
                std.debug.print("[pjrt-debug] requested={s}\n", .{canonical});
            }
            return error.PluginAlreadyLoaded;
        }
        cached_refcount += 1;
        if (debug) {
            std.debug.print("[pjrt-debug] load_plugin reuse: path={s} handle={*} refcount={}\n", .{
                existing,
                cached_api.?.handle,
                cached_refcount,
            });
        }
        std.heap.page_allocator.free(canonical);
        var api = cached_api.?;
        api.trace_execute = options.trace_execute;
        return api;
    }

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_z = try std.mem.printSentinel(&path_buf, "{s}", .{canonical}, 0);

    const handle = c.dlopen(path_z, c.RTLD_NOW | c.RTLD_GLOBAL) orelse {
        std.debug.print("dlopen failed: {s}\n", .{dl_err_msg()});
        return error.PluginLoadFailed;
    };

    const api = try load_from_handle(handle, canonical, options);
    cached_api = api;
    cached_path = canonical;
    cached_refcount = 1;
    return api;
}

fn maybe_dlopen(
    label: []const u8,
    soname: [:0]const u8,
    flags: c_int,
    verbose: bool,
) ?*anyopaque {
    _ = c.dlerror(); // clear
    const h = c.dlopen(soname.ptr, flags);
    if (h == null) {
        if (verbose) {
            std.debug.print("dlopen FAIL {s}: {s} -> {s}\n", .{ label, soname, dl_err_msg() });
        }
        return null;
    }
    if (verbose) {
        std.debug.print("dlopen OK   {s}: {s}\n", .{ label, soname });
    }
    return h;
}

const HostNvidiaHandles = struct {
    cuda: *anyopaque,
    nvml: ?*anyopaque,
};

const PreloadError = error{
    CudaDriverNotFound,
};

/// Preload host-injected NVIDIA driver libraries.
///
/// `libcuda.so.1` is required. NVML is loaded when available. The runtime
///  environment must make both libraries visible to the dynamic loader.
fn preload_host_nvidia(verbose: bool) PreloadError!HostNvidiaHandles {
    // PJRT plugin dependencies may resolve CUDA driver symbols globally.
    const flags_driver: c_int = c.RTLD_NOW | c.RTLD_GLOBAL;

    const cuda_h = maybe_dlopen("CUDA driver", "libcuda.so.1", flags_driver, verbose) orelse {
        if (!verbose) {
            std.debug.print("dlopen FAIL CUDA driver (libcuda.so.1) -> {s}\n", .{dl_err_msg()});
        }
        return error.CudaDriverNotFound;
    };

    const nvml_h = maybe_dlopen("NVML", "libnvidia-ml.so.1", flags_driver, verbose);

    return .{
        .cuda = cuda_h,
        .nvml = nvml_h,
    };
}

fn load_from_handle(
    handle: *anyopaque,
    canonical_path: []const u8,
    options: config.PluginOptions,
) !Api {
    errdefer _ = c.dlclose(handle);

    const get_api_sym = c.dlsym(handle, "GetPjrtApi") orelse {
        std.debug.print("dlsym(GetPjrtApi) failed: {s}\n", .{dl_err_msg()});
        return error.SymbolNotFound;
    };

    const get_api_fn: *const fn () callconv(.c) ?*const c.PJRT_Api =
        @ptrCast(@alignCast(get_api_sym));

    if (options.debug) {
        std.debug.print("[pjrt-debug] dlopen path={s} handle={*}\n", .{ canonical_path, handle });
        log_dladdr("GetPjrtApi", @ptrCast(@constCast(get_api_sym)), true);
    }

    var api = try Api.init(handle, get_api_fn, .{
        .trace_execute = options.trace_execute,
    });

    if (@field(api.pjrt_api, "PJRT_Plugin_Initialize")) |_| {
        var init_args = api_mod.init_args(c.PJRT_Plugin_Initialize_Args);
        try api.call("PJRT_Plugin_Initialize", &init_args);
    }

    if (options.debug) {
        std.debug.print("[pjrt-debug] PJRT_Api ptr={*} extension_start={*}\n", .{
            api.pjrt_api,
            api.pjrt_api.extension_start,
        });
        if (api.pjrt_api.extension_start) |ext_ptr| {
            log_dladdr("PJRT_Api extension_start", @ptrCast(@constCast(ext_ptr)), true);
        }
    }

    const ver = api.version();
    std.debug.print("PJRT api version from plugin: {}.{}\n", .{ ver.major, ver.minor });
    std.debug.print("sizeof(PJRT_Client_Create_Args) = {}\n", .{@sizeOf(c.PJRT_Client_Create_Args)});

    if (ver.major == 0 and ver.minor < 40) {
        std.debug.print(
            "Warning: PJRT API version {}.{} is old (expected >= 0.40)\n",
            .{ ver.major, ver.minor },
        );
    }

    return api;
}

/// Release one reference to a PJRT plugin.
///
/// The default retains the DSO for process lifetime because plugin global state
///  may outlive the backend instance and still reference DSO code or data.
pub fn unload_plugin(api: Api, options: config.PluginOptions) void {
    if (cached_api) |cached| {
        if (cached.handle == api.handle) {
            if (cached_refcount > 0) cached_refcount -= 1;
            if (options.debug) {
                std.debug.print("[pjrt-debug] unload_plugin handle={*} refcount={}\n", .{
                    api.handle,
                    cached_refcount,
                });
            }
            if (cached_refcount == 0 and options.close_on_unload) {
                _ = c.dlclose(api.handle);
                if (cached_path) |p| std.heap.page_allocator.free(p);
                cached_api = null;
                cached_path = null;
            }
            return;
        }
    }
    if (options.close_on_unload) _ = c.dlclose(api.handle);
}
