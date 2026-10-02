//! Convenience utility for managing cache paths.
//!
//! Root resolution: explicit root > `ZG_CACHE_DIR` env var > `/tmp/zigrad-cache`.
const std = @import("std");

const log = std.log.scoped(.@"zg/cache");

pub const Cache = struct {
    buf: [std.fs.max_path_bytes]u8,
    len: usize,

    pub const Options = struct {
        /// Create the directory if it doesn't exist.
        create: bool = true,
    };

    pub const InitOptions = struct {
        /// Explicit root path. If null, uses `ZG_CACHE_DIR` env var,
        ///  falling back to `/tmp/zigrad-cache`.
        root: ?[]const u8 = null,
        /// Create the root directory if it doesn't exist.
        create: bool = true,
    };

    /// Initialize a cache rooted at the given path, `ZG_CACHE_DIR`, or `/tmp/zigrad-cache`.
    pub fn init(
        /// I/O context used to create the root directory.
        io: std.Io,
        /// Environment used when `opts.root` is null.
        environ_map: *const std.process.Environ.Map,
        /// Root selection and directory creation policy.
        opts: InitOptions,
    ) !Cache {
        const root: []const u8 = opts.root orelse
            environ_map.get("ZG_CACHE_DIR") orelse
            "/tmp/zigrad-cache";
        if (root.len >= std.fs.max_path_bytes) return error.NameTooLong;
        var self: Cache = .{ .buf = undefined, .len = root.len };
        @memcpy(self.buf[0..root.len], root);
        if (opts.create) {
            std.Io.Dir.cwd().createDirPath(io, self.path()) catch |err| {
                log.err("failed to create cache root '{s}': {s}", .{ self.path(), @errorName(err) });
                return err;
            };
        }
        log.info("cache root: {s}", .{self.path()});
        return self;
    }

    /// The path this cache points to as a borrowed slice.
    pub fn path(self: *const Cache) []const u8 {
        return self.buf[0..self.len];
    }

    /// Null-terminated path as a borrowed slice for C interop.
    ///
    /// Writes a sentinel byte into the buffer.
    pub fn pathZ(self: *Cache) [:0]u8 {
        self.buf[self.len] = 0;
        return self.buf[0..self.len :0];
    }

    /// Append a path segment, optionally creating the resulting directory.
    pub fn subdir(self: *const Cache, io: std.Io, name: []const u8, opts: Options) !Cache {
        const result = try self.join(name);
        if (opts.create) {
            std.Io.Dir.cwd().createDirPath(io, result.path()) catch |err| {
                log.err("failed to create dir '{s}': {s}", .{ result.path(), @errorName(err) });
                return err;
            };
        }
        return result;
    }

    /// Append a path segment, nothing is created on the fs.
    pub fn join(self: *const Cache, name: []const u8) error{NameTooLong}!Cache {
        var result = self.*;
        const new_len = self.len + 1 + name.len;
        // Reserve one byte for a potential sentinel.
        if (new_len >= result.buf.len) return error.NameTooLong;
        result.buf[self.len] = '/';
        @memcpy(result.buf[self.len + 1 ..][0..name.len], name);
        result.len = new_len;
        return result;
    }
};

test Cache {
    const io = std.testing.io;
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    // init with explicit root (no env var dependency)
    const cache = try Cache.init(io, &environ, .{ .root = "/tmp/zigrad-cache-test" });
    try std.testing.expectEqualStrings("/tmp/zigrad-cache-test", cache.path());

    // subdir composes
    const tvm = try cache.subdir(io, "tvm", .{});
    try std.testing.expectEqualStrings("/tmp/zigrad-cache-test/tvm", tvm.path());

    // join composes without creating dirs
    const artifact = try tvm.join("kernel.so");
    try std.testing.expectEqualStrings("/tmp/zigrad-cache-test/tvm/kernel.so", artifact.path());

    // chain
    const cache_tvm = try cache.subdir(io, "tvm", .{});
    const deep = try cache_tvm.join("abc123");
    try std.testing.expectEqualStrings("/tmp/zigrad-cache-test/tvm/abc123", deep.path());

    // dirZ
    var z = try tvm.join("workload.json");
    const sentinel = z.pathZ();
    try std.testing.expectEqualStrings("/tmp/zigrad-cache-test/tvm/workload.json", sentinel);
    try std.testing.expectEqual(@as(u8, 0), sentinel[sentinel.len]);
}
