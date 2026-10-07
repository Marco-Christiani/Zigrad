//! Shared POSIX dynamic-library loader.

const std = @import("std");

const rtld_now: c_int = 0x2;
const rtld_global: c_int = 0x100;

extern "c" fn dlopen(filename: [*:0]const u8, flags: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: *anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlclose(handle: *anyopaque) c_int;
extern "c" fn dlerror() ?[*:0]const u8;

pub const Error = std.mem.Allocator.Error || error{
    OpenFailed,
};

pub const Visibility = enum {
    local,
    global,
};

pub const OpenOptions = struct {
    visibility: Visibility = .local,
};

/// Dynamic-library handle closed by `deinit`.
pub const Library = struct {
    handle: *anyopaque,

    /// Open one explicit path with immediate symbol resolution.
    pub fn open(
        allocator: std.mem.Allocator,
        path: []const u8,
        options: OpenOptions,
    ) Error!Library {
        const path_z = try allocator.dupeSentinel(u8, path, 0);
        defer allocator.free(path_z);

        _ = dlerror();
        const visibility: c_int = switch (options.visibility) {
            .local => 0,
            .global => rtld_global,
        };
        const handle = dlopen(path_z.ptr, rtld_now | visibility) orelse
            return error.OpenFailed;
        return .{ .handle = handle };
    }

    /// Resolve one symbol as the requested pointer type.
    pub fn lookup(
        self: Library,
        comptime Pointer: type,
        name: [:0]const u8,
    ) ?Pointer {
        _ = dlerror();
        const address = dlsym(self.handle, name.ptr) orelse return null;
        return @ptrCast(@alignCast(address));
    }

    /// Close the library handle.
    pub fn close(self: *Library) void {
        _ = dlclose(self.handle);
        self.* = undefined;
    }
};

/// Return the current thread's dynamic-loader diagnostic.
pub fn error_message() []const u8 {
    const message = dlerror() orelse return "dynamic loader returned no diagnostic";
    return std.mem.span(message);
}
