//! Special Operations
//! Ops with unique semantics (custom_call, call).
const std = @import("std");
const types = @import("types.zig");
const pr = @import("../pr.zig");
const Aval = pr.Aval;

// ============================================================================
// Custom Call
// ============================================================================

pub const custom_call = struct {
    pub const arity = .{ .in = .variadic, .out = .variadic };

    pub fn validate(_: *const pr.Op, _: pr.CustomCallParams) pr.ValidationError!void {}

    pub fn infer_output(_: std.mem.Allocator, _: []const *pr.Var, _: pr.CustomCallParams) pr.BuildError!Aval {
        // Custom-call result types are explicit at construction.
        return error.InvalidOpArity;
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, cc: pr.CustomCallParams) types.FormatError!void {
        try writer.print("target=\"{s}\"", .{cc.target_name});
        if (cc.has_side_effect) try writer.writeAll(", side_effect=true");
        if (cc.payload.len > 0) try writer.print(", payload=0x{x}", .{cc.payload});
    }
};

// ============================================================================
// Call
// ============================================================================

pub const call = struct {
    pub const arity = .{ .in = .variadic, .out = .variadic };

    pub fn validate(_: *const pr.Op, _: pr.CallParams) pr.ValidationError!void {}

    pub fn infer_output(_: std.mem.Allocator, _: []const *pr.Var, _: pr.CallParams) pr.BuildError!Aval {
        // Call result types come from the callee and are explicit at construction.
        return error.InvalidOpArity;
    }

    pub fn format(writer: *types.Writer, _: *const pr.Op, cp: pr.CallParams) types.FormatError!void {
        try writer.print("callee=@{d}", .{@backingInt(cp.callee)});
    }
};
