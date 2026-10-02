//! Process-wide TVM runtime-library lifecycle.
//!
//! Configuration and the first library load run during single-threaded setup.

const std = @import("std");

const dylib = @import("../c/dylib.zig");
const ffi = @import("../c/tvm/ffi.zig");
const raw = @import("../c/tvm/c.zig");
pub const Config = @import("config.zig").RuntimeConfig;
pub const Surface = @import("config.zig").RuntimeSurface;

const log = std.log.scoped(.@"zg/tvm_runtime_loader");

/// Errors returned when configuration changes after runtime initialization.
pub const ConfigureError = error{
    /// A configured library path exceeds the loader's path capacity.
    RuntimeLibraryPathTooLong,
    /// A different configuration was supplied after a library was loaded.
    RuntimeAlreadyInitialized,
};

/// Runtime errors exposed by the TVM integration.
pub const Error = ffi.TvmError || error{
    /// No TVM library configuration was supplied.
    TvmNotConfigured,
    /// The configured library does not provide the requested operation.
    TvmSurfaceUnavailable,
};

const State = struct {
    ffi: dylib.Library,
    surface: ?dylib.Library = null,
};

const LibraryPath = struct {
    bytes: [std.fs.max_path_bytes]u8,
    length: usize,

    fn set(self: *LibraryPath, path: []const u8) ConfigureError!void {
        if (path.len >= self.bytes.len) return error.RuntimeLibraryPathTooLong;
        @memcpy(self.bytes[0..path.len], path);
        self.length = path.len;
        std.debug.assert(std.mem.eql(u8, self.slice(), path));
    }

    fn slice(self: *const LibraryPath) []const u8 {
        std.debug.assert(self.length < self.bytes.len);
        return self.bytes[0..self.length];
    }
};

const RuntimeConfiguration = struct {
    surface: Surface,
    ffi: LibraryPath,
    runtime: LibraryPath,
    compiler: LibraryPath,

    fn set(self: *RuntimeConfiguration, next: Config) ConfigureError!void {
        // Validate every path before changing the active configuration.
        if (next.ffi.path.len >= std.fs.max_path_bytes)
            return error.RuntimeLibraryPathTooLong;
        if (next.runtime.path.len >= std.fs.max_path_bytes)
            return error.RuntimeLibraryPathTooLong;
        if (next.compiler.path.len >= std.fs.max_path_bytes)
            return error.RuntimeLibraryPathTooLong;

        self.surface = next.surface;
        try self.ffi.set(next.ffi.path);
        try self.runtime.set(next.runtime.path);
        try self.compiler.set(next.compiler.path);
    }

    fn eql(self: *const RuntimeConfiguration, other: Config) bool {
        return self.surface == other.surface and
            std.mem.eql(u8, self.ffi.slice(), other.ffi.path) and
            std.mem.eql(u8, self.runtime.slice(), other.runtime.path) and
            std.mem.eql(u8, self.compiler.slice(), other.compiler.path);
    }
};

var configured: ?RuntimeConfiguration = null;
var state: ?State = null;

/// Configure TVM before the first runtime operation.
///
/// Library paths are copied into fixed-capacity process storage.
/// Repeating the active configuration is idempotent after initialization.
pub fn configure(next: Config) ConfigureError!void {
    if (state != null) {
        const active = &configured.?;
        if (active.eql(next)) return;
        return error.RuntimeAlreadyInitialized;
    }
    var stored: RuntimeConfiguration = undefined;
    try stored.set(next);
    configured = stored;
}

/// Load the configured TVM runtime surface.
pub fn ensure_loaded(requirement: Surface) Error!void {
    const active = if (configured) |*value| value else return error.TvmNotConfigured;
    if (!active.surface.satisfies(requirement)) {
        log.err(
            "TVM {t} operation exceeds configured {t} surface",
            .{ requirement, active.surface },
        );
        return error.TvmSurfaceUnavailable;
    }

    try ensure_ffi_loaded();
    if (requirement != .ffi) try ensure_surface_loaded();
}

/// Return TVM global-function names allocated by `allocator` for diagnostics.
pub fn list_global_names(allocator: std.mem.Allocator, requirement: Surface) Error![][]const u8 {
    try ensure_loaded(requirement);
    return try ffi.list_global_names(allocator);
}

fn ensure_ffi_loaded() Error!void {
    if (state != null) return;
    const active = if (configured) |*value| value else return error.TvmNotConfigured;

    var library = dylib.Library.open(std.heap.smp_allocator, active.ffi.slice(), .{
        .visibility = .global,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.OpenFailed => {
            log.err(
                "failed to open TVM FFI library '{s}': {s}",
                .{ active.ffi.slice(), dylib.error_message() },
            );
            return error.TvmLoadFailed;
        },
    };
    errdefer library.close();

    raw.install_symbols(library) catch |err| {
        log.err("failed to resolve TVM FFI symbols: {s}", .{@errorName(err)});
        return error.TvmLoadFailed;
    };

    state = .{ .ffi = library };
    log.info("loaded TVM FFI library '{s}'", .{active.ffi.slice()});
}

fn ensure_surface_loaded() Error!void {
    if (state.?.surface != null) return;
    const active = if (configured) |*value| value else return error.TvmNotConfigured;

    const selected = switch (active.surface) {
        .ffi => unreachable,
        .runtime => &active.runtime,
        .compiler => &active.compiler,
    };

    const library = dylib.Library.open(std.heap.smp_allocator, selected.slice(), .{
        .visibility = .global,
    }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        error.OpenFailed => {
            log.err(
                "failed to open TVM {t} library '{s}': {s}",
                .{ active.surface, selected.slice(), dylib.error_message() },
            );
            return error.TvmLoadFailed;
        },
    };

    state.?.surface = library;
    log.info("loaded TVM {t} library '{s}'", .{ active.surface, selected.slice() });
}

test "Config uses the configured TVM sonames" {
    try std.testing.expectEqualStrings(
        @import("config.zig").default_ffi_path,
        (Config{ .surface = .ffi }).ffi.path,
    );
    try std.testing.expectEqualStrings(
        @import("config.zig").default_runtime_path,
        (Config{ .surface = .runtime }).runtime.path,
    );
    try std.testing.expectEqualStrings(
        @import("config.zig").default_compiler_path,
        (Config{ .surface = .compiler }).compiler.path,
    );
}

test "RuntimeConfiguration copies library paths" {
    var ffi_path = [_]u8{ 'f', 'f', 'i' };
    var runtime_path = [_]u8{ 'r', 'u', 'n' };
    var compiler_path = [_]u8{ 'c', 'o', 'm', 'p', 'i', 'l', 'e' };
    var stored: RuntimeConfiguration = undefined;
    try stored.set(.{
        .surface = .compiler,
        .ffi = .{ .path = &ffi_path },
        .runtime = .{ .path = &runtime_path },
        .compiler = .{ .path = &compiler_path },
    });

    @memset(&ffi_path, 'x');
    @memset(&runtime_path, 'x');
    @memset(&compiler_path, 'x');

    try std.testing.expectEqualStrings("ffi", stored.ffi.slice());
    try std.testing.expectEqualStrings("run", stored.runtime.slice());
    try std.testing.expectEqualStrings("compile", stored.compiler.slice());
}
