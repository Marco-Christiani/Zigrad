//! TVM runtime container bindings.

const std = @import("std");
const ffi = @import("ffi.zig");
const c = @import("c.zig");

const Value = ffi.Value;

/// TVM `ffi.Array`.
pub const Array = struct {
    object: ffi.Object,

    /// Take a packed result as an array.
    pub fn take(value: *ffi.OwnedValue) !Array {
        try value.borrow().require_type_index(c.kTVMFFIArray);
        return .{ .object = try .take(value) };
    }

    /// Retain `value` as an array.
    pub fn retain(value: Value) !Array {
        try value.require_type_index(c.kTVMFFIArray);
        const object = try ffi.Object.retain(value);
        return .{ .object = object };
    }

    /// Construct an array from packed values.
    pub fn from_values(allocator: std.mem.Allocator, items: []const Value) !Array {
        return try ffi.call_global_take(Array, allocator, "ffi.Array", items);
    }

    /// Return the number of array elements.
    pub fn len(self: Array, allocator: std.mem.Allocator) !usize {
        var result = try ffi.call_global(
            allocator,
            "ffi.ArraySize",
            &.{self.as_value()},
        );
        defer result.deinit();
        const count = try result.borrow().require_int();
        return std.math.cast(usize, count) orelse error.IntegerOutOfRange;
    }

    /// Copy the element at `index` into a packed value.
    pub fn get(
        self: Array,
        allocator: std.mem.Allocator,
        index: usize,
    ) !ffi.OwnedValue {
        const packed_index = std.math.cast(i64, index) orelse
            return error.IntegerOutOfRange;
        return try ffi.call_global(allocator, "ffi.ArrayGetItem", &.{
            self.as_value(), Value.int(packed_index),
        });
    }

    pub fn as_value(self: *const Array) Value {
        return self.object.as_value();
    }

    /// Remove the packed value from this wrapper.
    pub fn take_value(self: *Array) ffi.OwnedValue {
        return self.object.take_value();
    }

    pub fn deinit(self: *Array) void {
        self.object.deinit();
        self.* = undefined;
    }
};

/// TVM `ffi.Map`.
pub const Map = struct {
    object: ffi.Object,

    /// Take a packed result as a map.
    pub fn take(value: *ffi.OwnedValue) !Map {
        try value.borrow().require_type_index(c.kTVMFFIMap);
        return .{ .object = try .take(value) };
    }

    /// Retain `value` as a map.
    pub fn retain(value: Value) !Map {
        try value.require_type_index(c.kTVMFFIMap);
        const object = try ffi.Object.retain(value);
        return .{ .object = object };
    }

    /// Construct a map from alternating key and value entries.
    pub fn from_pairs(allocator: std.mem.Allocator, pairs: []const Value) !Map {
        if (pairs.len % 2 != 0) return error.InvalidMapEntries;
        return try ffi.call_global_take(Map, allocator, "ffi.Map", pairs);
    }

    /// Return the number of map entries.
    pub fn len(self: Map, allocator: std.mem.Allocator) !usize {
        var result = try ffi.call_global(
            allocator,
            "ffi.MapSize",
            &.{self.as_value()},
        );
        defer result.deinit();
        const count = try result.borrow().require_int();
        return std.math.cast(usize, count) orelse error.IntegerOutOfRange;
    }

    /// Return whether `key` is present.
    pub fn contains(self: Map, allocator: std.mem.Allocator, key: Value) !bool {
        var result = try ffi.call_global(
            allocator,
            "ffi.MapCount",
            &.{ self.as_value(), key },
        );
        defer result.deinit();
        const count = try result.borrow().require_int();
        return count != 0;
    }

    /// Copy the value associated with `key` into a packed value.
    pub fn get(
        self: Map,
        allocator: std.mem.Allocator,
        key: Value,
    ) !ffi.OwnedValue {
        return try ffi.call_global(
            allocator,
            "ffi.MapGetItem",
            &.{ self.as_value(), key },
        );
    }

    pub fn as_value(self: *const Map) Value {
        return self.object.as_value();
    }

    pub fn deinit(self: *Map) void {
        self.object.deinit();
        self.* = undefined;
    }
};
