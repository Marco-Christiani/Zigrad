//! Source-byte identity for the configured TVM lowering module graph.
const std = @import("std");

const roots = [_][]const u8{
    "src/tvm/matmul.zig",
    "src/c/tvm/compile.zig",
    "src/tvm/tune.zig",
    "src/tvm/export.zig",
    "src/tvm/nvrtc_callback.zig",
    "src/cuda/nvrtc.zig",
};

pub const Identity = struct { digest: []const u8, manifest: []const u8 };
const Input = struct { path: []const u8, bytes: []const u8 };
const Walk = struct {
    b: *std.Build,
    visited: std.StringHashMap(void),
    inputs: std.StringHashMap([]const u8),
    edges: std.ArrayList([]const u8) = .empty,
    inactive_modules: []const []const u8,

    fn edge(self: *Walk, path: []const u8, imported: []const u8, classification: []const u8) !void {
        try self.edges.append(self.b.allocator, self.b.fmt("{s}\t{s}\t{s}\n", .{ path, imported, classification }));
    }

    fn visit(self: *Walk, module: *std.Build.Module, path: []const u8) anyerror!void {
        const b = self.b;
        const key = b.fmt("{x}:{s}", .{ @intFromPtr(module), path });
        if (self.visited.contains(key)) return;
        try self.visited.put(key, {});
        const bytes = try std.Io.Dir.cwd().readFileAlloc(b.graph.io, b.pathFromRoot(path), b.allocator, .limited(4 * 1024 * 1024));
        try self.inputs.put(path, bytes);
        const source = try b.allocator.dupeZ(u8, bytes);
        var tokenizer = std.zig.Tokenizer.init(source);
        while (true) {
            const token = tokenizer.next();
            if (token.tag == .eof) break;
            if (token.tag != .builtin) continue;
            const builtin = source[token.loc.start..token.loc.end];
            const embedded = std.mem.eql(u8, builtin, "@embedFile");
            if (!embedded and !std.mem.eql(u8, builtin, "@import")) continue;
            if (tokenizer.next().tag != .l_paren) return error.InvalidImport;
            const literal = tokenizer.next();
            if (literal.tag != .string_literal) return error.UnresolvedDynamicImport;
            const imported = try std.zig.string_literal.parseAlloc(b.allocator, source[literal.loc.start..literal.loc.end]);
            if (embedded or std.mem.endsWith(u8, imported, ".zig")) {
                const absolute = try std.fs.path.resolve(b.allocator, &.{ b.pathFromRoot(std.fs.path.dirname(path) orelse "."), imported });
                const relative = try std.fs.path.relative(b.allocator, b.build_root.path orelse ".", null, b.build_root.path orelse ".", absolute);
                if (std.mem.startsWith(u8, relative, "..")) return error.SourceOutsideRepository;
                try self.edge(path, imported, b.fmt("{s}:{s}", .{ if (embedded) "embedded" else "relative", relative }));
                if (embedded) {
                    const data = try std.Io.Dir.cwd().readFileAlloc(b.graph.io, absolute, b.allocator, .limited(64 * 1024 * 1024));
                    try self.inputs.put(relative, data);
                } else try self.visit(module, relative);
                continue;
            }
            if (std.mem.eql(u8, imported, "std") or std.mem.eql(u8, imported, "builtin")) {
                try self.edge(path, imported, "toolchain");
            } else if (std.mem.eql(u8, imported, "build_options")) {
                try self.edge(path, imported, "generated:configured-options");
            } else if (module.import_table.get(imported)) |forwarded| {
                const root = forwarded.root_source_file orelse return error.MissingModuleRoot;
                switch (root) {
                    .generated => try self.edge(path, imported, "external:translated-C"),
                    else => {
                        const absolute = root.getPath(b);
                        const relative = try std.fs.path.relative(b.allocator, b.build_root.path orelse ".", null, b.build_root.path orelse ".", absolute);
                        if (std.mem.startsWith(u8, relative, "..")) {
                            // Dependency roots are pinned by the package graph.
                            try self.edge(path, imported, "external:package");
                        } else {
                            try self.edge(path, imported, b.fmt("named:{s}", .{relative}));
                            try self.visit(forwarded, relative);
                        }
                    },
                }
            } else if (contains(self.inactive_modules, imported)) {
                // Absent translated modules are unavailable integrations.
                try self.edge(path, imported, "inactive:translated-C");
            } else {
                std.debug.print("unresolved TVM source edge: {s} -> {s}\n", .{ path, imported });
                return error.UnresolvedModule;
            }
        }
    }
};

/// Resolve imports through the build-owned graph and hash current source bytes.
///  Paths in the manifest are relative to the checkout, so relocation preserves identity.
pub fn derive(b: *std.Build, module: *std.Build.Module, configuration: []const u8, inactive_modules: []const []const u8) !Identity {
    var walk = Walk{ .b = b, .visited = std.StringHashMap(void).init(b.allocator), .inputs = std.StringHashMap([]const u8).init(b.allocator), .inactive_modules = inactive_modules };
    for (roots) |root| try walk.visit(module, root);
    var inputs: std.ArrayList(Input) = .empty;
    var iterator = walk.inputs.iterator();
    while (iterator.next()) |entry| try inputs.append(b.allocator, .{ .path = entry.key_ptr.*, .bytes = entry.value_ptr.* });
    std.mem.sort(Input, inputs.items, {}, struct {
        fn less(_: void, a: Input, c: Input) bool {
            return std.mem.lessThan(u8, a.path, c.path);
        }
    }.less);
    std.mem.sort([]const u8, walk.edges.items, {}, struct {
        fn less(_: void, a: []const u8, c: []const u8) bool {
            return std.mem.lessThan(u8, a, c);
        }
    }.less);
    var hashing = std.crypto.hash.Blake3.init(.{});
    hashing.update("zigrad.tvm.lowering.v1\x00");
    hashing.update(configuration);
    hashing.update(@import("builtin").zig_version_string);
    var manifest: std.Io.Writer.Allocating = .init(b.allocator);
    try manifest.writer.print("configuration\t{s}\nzig\t{s}\n", .{ configuration, @import("builtin").zig_version_string });
    for (inputs.items) |input| {
        var content: [32]u8 = undefined;
        std.crypto.hash.Blake3.hash(input.bytes, &content, .{});
        const hex = std.fmt.bytesToHex(content, .lower);
        hashing.update(input.path);
        hashing.update(&.{0});
        hashing.update(&content);
        try manifest.writer.print("source\t{s}\t{s}\n", .{ input.path, hex });
    }
    for (walk.edges.items) |edge| {
        hashing.update(edge);
        try manifest.writer.writeAll(edge);
    }
    var digest: [32]u8 = undefined;
    hashing.final(&digest);
    const identity = b.dupe(&std.fmt.bytesToHex(digest, .lower));
    try manifest.writer.print("identity\t{s}\n", .{identity});
    return .{ .digest = identity, .manifest = try manifest.toOwnedSlice() };
}

fn contains(names: []const []const u8, name: []const u8) bool {
    for (names) |candidate| if (std.mem.eql(u8, candidate, name)) return true;
    return false;
}
