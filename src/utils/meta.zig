//! Comptime operations for nested structs with a designated leaf type.
//!
//! `Tree` and callable binding use the same declaration-order DFS traversal.
const std = @import("std");

/// Count leaves of type `Leaf` in struct type `T` at comptime.
pub fn leaf_count(comptime Leaf: type, comptime T: type) comptime_int {
    if (T == Leaf) return 1;
    return switch (@typeInfo(T)) {
        .@"struct" => |info| blk: {
            var total: comptime_int = 0;
            for (info.field_types) |field_type| total += leaf_count(Leaf, field_type);
            break :blk total;
        },
        .array => |info| info.len * leaf_count(Leaf, info.child),
        else => @compileError(std.fmt.comptimePrint(
            "{s}: unsupported type `{s}` expected {s}, struct, or array",
            .{ @src().fn_name, @typeName(T), @typeName(Leaf) },
        )),
    };
}

test leaf_count {
    const Inner = struct { a: i32, b: i32 };
    const Outer = struct { x: Inner, y: i32 };

    try std.testing.expectEqual(3, leaf_count(i32, Outer));
    try std.testing.expectEqual(1, leaf_count(i32, i32));
    try std.testing.expectEqual(2, leaf_count(Inner, struct { a: Inner, b: Inner }));
}

/// Flatten a struct value into a pre-allocated leaf array (DFS order).
pub fn flatten(comptime Leaf: type, comptime T: type, value: T, out: []Leaf, idx: *usize) void {
    if (T == Leaf) {
        out[idx.*] = value;
        idx.* += 1;
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.field_names, info.field_types) |field_name, field_type| {
                flatten(Leaf, field_type, @field(value, field_name), out, idx);
            }
        },
        .array => |info| {
            inline for (0..info.len) |i| {
                flatten(Leaf, info.child, value[i], out, idx);
            }
        },
        else => @compileError(std.fmt.comptimePrint(
            "{s}: unsupported type `{s}` expected {s}, struct, or array",
            .{ @src().fn_name, @typeName(T), @typeName(Leaf) },
        )),
    }
}

test flatten {
    {
        const Inner = struct { a: i32, b: i32 };
        const S = struct { x: Inner, y: i32 };

        const input = S{ .x = .{ .a = 10, .b = 20 }, .y = 30 };

        var buf: [3]i32 = undefined;
        var idx: usize = 0;
        flatten(i32, S, input, &buf, &idx);

        try std.testing.expectEqual(3, idx);
        try std.testing.expectEqual(10, buf[0]);
        try std.testing.expectEqual(20, buf[1]);
        try std.testing.expectEqual(30, buf[2]);

        idx = 0;
        const recovered = unflatten(i32, S, &buf, &idx);
        try std.testing.expectEqual(10, recovered.x.a);
        try std.testing.expectEqual(20, recovered.x.b);
        try std.testing.expectEqual(30, recovered.y);
    }
    {
        // Array fields
        const S = struct { vals: [3]i32 };

        try std.testing.expectEqual(3, leaf_count(i32, S));

        const input = S{ .vals = .{ 1, 2, 3 } };
        var buf: [3]i32 = undefined;
        var idx: usize = 0;
        flatten(i32, S, input, &buf, &idx);
        try std.testing.expectEqual(1, buf[0]);
        try std.testing.expectEqual(3, buf[2]);
    }
}

/// Reconstruct a struct value from a flat leaf array (DFS order).
///
/// Returns `RuntimeOf(T)` so comptime fields (e.g. from module-level
/// `const` values in anonymous tuples) can be assigned at runtime.
pub fn unflatten(comptime Leaf: type, comptime T: type, leaves: []const Leaf, idx: *usize) RuntimeOf(T) {
    if (T == Leaf) {
        const val = leaves[idx.*];
        idx.* += 1;
        return val;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            var result: RuntimeOf(T) = undefined;
            inline for (info.field_names, info.field_types) |field_name, field_type| {
                @field(result, field_name) = unflatten(Leaf, field_type, leaves, idx);
            }
            return result;
        },
        .array => |info| {
            var result: RuntimeOf(T) = undefined;
            inline for (0..info.len) |i| {
                result[i] = unflatten(Leaf, info.child, leaves, idx);
            }
            return result;
        },
        else => @compileError(std.fmt.comptimePrint(
            "{s}: unsupported type `{s}` expected {s}, struct, or array",
            .{ @src().fn_name, @typeName(T), @typeName(Leaf) },
        )),
    }
}

test unflatten {
    // check unflatten handles comptime fields via RuntimeOf
    const Inner = struct { a: i32, b: i32 };
    const c1: Inner = .{ .a = 1, .b = 2 };
    const c2: Inner = .{ .a = 3, .b = 4 };
    const comptime_tuple = .{ c1, c2 };

    var buf = [_]i32{ 10, 20, 30, 40 };
    var idx: usize = 0;
    const recovered = unflatten(i32, @TypeOf(comptime_tuple), &buf, &idx);
    try std.testing.expectEqual(10, recovered.@"0".a);
    try std.testing.expectEqual(40, recovered.@"1".b);
}

/// Walk a struct value by pointer, calling `f` on each leaf.
///
/// `f` receives `*Leaf` and may modify or release leaves in place.
/// Use `flatten` for read-only traversal.
pub fn visit(comptime Leaf: type, comptime T: type, target: *T, comptime f: fn (*Leaf) void) void {
    if (T == Leaf) {
        f(target);
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.field_names, info.field_types) |field_name, field_type| {
                visit(Leaf, field_type, &@field(target, field_name), f);
            }
        },
        .array => |info| {
            inline for (0..info.len) |i| {
                visit(Leaf, info.child, &target[i], f);
            }
        },
        else => @compileError(std.fmt.comptimePrint(
            "{s}: unsupported type `{s}` expected {s}, struct, or array",
            .{ @src().fn_name, @typeName(T), @typeName(Leaf) },
        )),
    }
}

test visit {
    const Inner = struct { a: i32, b: i32 };
    const S = struct { x: Inner, y: i32 };

    var value = S{ .x = .{ .a = 1, .b = 2 }, .y = 3 };
    visit(i32, S, &value, struct {
        fn f(leaf: *i32) void {
            leaf.* *= 10;
        }
    }.f);

    try std.testing.expectEqual(10, value.x.a);
    try std.testing.expectEqual(20, value.x.b);
    try std.testing.expectEqual(30, value.y);
}

/// Make struct fields writable at runtime.
/// Module constants in anonymous tuples can infer comptime fields.
/// TODO(api): Reject implicit comptime fields when callers can avoid them directly.
pub fn RuntimeOf(comptime T: type) type {
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            for (info.field_attrs) |attrs| {
                if (!attrs.@"comptime") continue;
                if (info.is_tuple) return @Tuple(info.field_types);
                var field_attrs: [info.field_names.len]std.lang.Type.Struct.FieldAttributes = undefined;
                for (info.field_attrs, 0..) |field, i| field_attrs[i] = .{ .@"align" = field.@"align" };
                return @Struct(info.layout, null, info.field_names, info.field_types, &field_attrs);
            }
            return T;
        },
        else => return T,
    }
}

// Path generation

/// Generate dot-separated paths for all leaves in a struct type.
///
/// Returns a comptime array of string literals.
pub fn tree_paths(comptime Leaf: type, comptime T: type) [leaf_count(Leaf, T)][]const u8 {
    @setEvalBranchQuota(100000);
    var result: [leaf_count(Leaf, T)][]const u8 = undefined;
    var idx: usize = 0;
    build_paths(Leaf, T, &result, &idx, "");
    return result;
}

test tree_paths {
    const Layer = struct { w: u8, b: u8 };
    const Model = struct {
        embed: u8,
        layers: [2]Layer,
        out: u8,
    };

    const paths = tree_paths(u8, Model);
    try std.testing.expectEqual(6, paths.len);
    try std.testing.expectEqualStrings("embed", paths[0]);
    try std.testing.expectEqualStrings("layers.0.w", paths[1]);
    try std.testing.expectEqualStrings("layers.0.b", paths[2]);
    try std.testing.expectEqualStrings("layers.1.w", paths[3]);
    try std.testing.expectEqualStrings("layers.1.b", paths[4]);
    try std.testing.expectEqualStrings("out", paths[5]);
}

fn build_paths(
    comptime Leaf: type,
    comptime T: type,
    result: [][]const u8,
    idx: *usize,
    comptime prefix: []const u8,
) void {
    if (T == Leaf) {
        result[idx.*] = prefix;
        idx.* += 1;
        return;
    }
    switch (@typeInfo(T)) {
        .@"struct" => |info| {
            inline for (info.field_names, info.field_types) |field_name, field_type| {
                const sep = if (prefix.len == 0) "" else ".";
                build_paths(Leaf, field_type, result, idx, prefix ++ sep ++ field_name);
            }
        },
        .array => |info| {
            inline for (0..info.len) |i| {
                const sep = if (prefix.len == 0) "" else ".";
                build_paths(Leaf, info.child, result, idx, prefix ++ sep ++ std.fmt.comptimePrint("{d}", .{i}));
            }
        },
        else => @compileError(std.fmt.comptimePrint(
            "{s}: unsupported type `{s}` expected {s}, struct, or array",
            .{ @src().fn_name, @typeName(T), @typeName(Leaf) },
        )),
    }
}
