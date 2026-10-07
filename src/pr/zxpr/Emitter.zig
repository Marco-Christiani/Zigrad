//! ZXPR: Zigrad Expression Representation
//!
//! A human-readable text format for zigrad's Program Representation (PR).
//! Inspired by jaxpr.
//!
//! Example:
//! ```zxpr
//! zxpr main {
//!   ; params (2)
//!   %0: 2x3<f32>
//!   %1: 3x2<f32>
//!   ; body (1 ops)
//!   let
//!     %2: 2x2<f32> = mm[M=2, K=3, N=2](%0, %1)  ; vjp
//!   in %2
//! }
//! ```
const std = @import("std");
const pr = @import("../pr.zig");
const ops = @import("../ops/ops.zig");
const style = @import("style.zig");

const Writer = std.Io.Writer;

const max_region_stack = 8;

const Self = @This();

writer: *Writer,
func: pr.Function,
function_id: ?pr.FunctionId,
indent: []const u8,
styler: style.Styler,

pub fn init(writer: *Writer, func: pr.Function, function_id: ?pr.FunctionId, cfg: style.Config) Self {
    return .{
        .writer = writer,
        .func = func,
        .function_id = function_id,
        .indent = "  ",
        .styler = style.Styler.init(writer, cfg),
    };
}

pub fn emit(self: *Self) !void {
    const w = self.writer;
    const ind = self.indent;

    try self.styler.write_keyword("zxpr");
    try w.writeAll(" ");
    if (self.function_id) |id| {
        try w.print("@{d} ", .{@backingInt(id)});
    }
    try self.styler.write_identifier(self.func.name);
    if (self.func.annotations.len > 0) {
        try self.styler.write_region("[");
        try self.emit_annotations(self.func.annotations);
        try self.styler.write_region("]");
    }
    try w.writeAll(" {\n");

    // Parameters section
    if (self.func.params.len > 0) {
        try w.writeAll(ind);
        try self.styler.write_section("; params");
        try w.print(" ({d})\n", .{self.func.params.len});
        for (self.func.params) |param_var| {
            try w.writeAll(ind);
            try self.emit_var_with_type(param_var);
            try w.writeAll("\n");
        }
    }

    // Body section
    if (self.func.ops.len > 0) {
        try w.writeAll(ind);
        try self.styler.write_section("; body");
        try w.print(" ({d} ops)\n", .{self.func.ops.len});
        try w.writeAll(ind);
        try self.styler.write_keyword("let");
        try w.writeAll("\n");

        // Track which regions are currently open (by index into func.regions)
        var open_stack: [max_region_stack]usize = undefined;
        var open_len: usize = 0;

        for (self.func.ops, 0..) |op, op_idx| {
            // Close regions that end before this op
            while (open_len > 0) {
                const ri = open_stack[open_len - 1];
                const region = self.func.regions[ri];
                const end_idx = region_end_index(self.func, region) orelse 0;
                if (op_idx > end_idx) {
                    open_len -= 1;
                    try self.emit_region_end(open_len);
                } else break;
            }

            // Open regions that start at this op
            for (self.func.regions, 0..) |region, ri| {
                const start_idx = region_start_index(self.func, region) orelse continue;
                if (start_idx == op_idx) {
                    // Check not already open
                    var already = false;
                    for (open_stack[0..open_len]) |oi| {
                        if (oi == ri) {
                            already = true;
                            break;
                        }
                    }
                    if (!already and open_len < max_region_stack) {
                        try self.emit_region_start(open_len, region);
                        open_stack[open_len] = ri;
                        open_len += 1;
                    }
                }
            }

            try w.writeAll(ind);
            try w.writeAll(ind);
            try self.emit_gutters(open_len);
            try self.emit_binding(op);
            try w.writeAll("\n");
        }

        // Close remaining open regions
        while (open_len > 0) {
            open_len -= 1;
            try self.emit_region_end(open_len);
        }
    }

    // Return section
    if (self.func.returns.len > 0) {
        if (self.func.ops.len == 0) {
            try w.writeAll(ind);
            try w.writeAll(";\n");
        }
        try w.writeAll(ind);
        try self.styler.write_keyword("in");
        try w.writeAll(" ");
        for (self.func.returns, 0..) |ret_var, i| {
            if (i > 0) try w.writeAll(", ");
            try self.emit_value_id(ret_var);
        }
        try w.writeAll("\n");
    }

    try w.writeAll("}\n");
}

fn region_start_index(func: pr.Function, region: pr.Region) ?usize {
    if (region.op_ids.len == 0) return null;
    return func.op_index_by_id(region.op_ids[0]);
}

fn region_end_index(func: pr.Function, region: pr.Region) ?usize {
    if (region.op_ids.len == 0) return null;
    return func.op_index_by_id(region.op_ids[region.op_ids.len - 1]);
}

fn emit_value_id(self: *Self, v: *const pr.Var) !void {
    try self.styler.write_value_id(v.id);
}

pub fn emit_var_with_type(self: *Self, v: *const pr.Var) !void {
    try self.emit_value_id(v);
    try self.writer.writeAll(": ");
    try self.emit_type(v.aval);
}

fn emit_type(self: *Self, aval: pr.Aval) !void {
    switch (aval) {
        .tensor => |t| {
            switch (self.styler.cfg.shape_format) {
                .dtype_suffix => {
                    if (t.shape.dims.len > 0) {
                        for (t.shape.dims, 0..) |d, i| {
                            if (i > 0) try self.writer.writeAll("x");
                            try self.writer.print("{d}", .{d});
                        }
                        try self.writer.writeAll("<");
                        try self.styler.write_type_name(@tagName(t.dtype));
                        try self.writer.writeAll(">");
                    } else {
                        try self.styler.write_type_name(@tagName(t.dtype));
                    }
                },
            }
        },
    }
}

pub fn emit_binding(self: *Self, op: *const pr.Op) !void {
    // Output binding(s)
    for (op.outputs, 0..) |out_var, i| {
        if (i > 0) try self.writer.writeAll(", ");
        try self.emit_var_with_type(out_var);
    }

    const prim = op.prim();
    try self.writer.writeAll(" = ");
    try self.styler.write_op_name(@tagName(prim));
    if (is_dtype_only_attr(prim) and !self.styler.cfg.include_dtype_attrs) {
        try self.writer.writeAll("(");
    } else {
        try self.writer.writeAll("[");
        if (is_dtype_only_attr(prim)) {
            _ = try self.emit_dtype_attr(op);
        } else {
            try ops.format(self.writer, op);
        }
        try self.writer.writeAll("](");
    }

    // Inputs as function args
    for (op.inputs, 0..) |operand, i| {
        if (i > 0) try self.writer.writeAll(", ");
        try self.emit_value_id(operand.value);
    }

    try self.writer.writeAll(")");

    // VJP/JVP annotation
    const has_vjp = ops.has_local_vjp(prim);
    const has_jvp = ops.has_local_jvp(prim);
    if (!has_vjp or !has_jvp) {
        try self.styler.write_comment("  ;");
        if (!has_jvp) try self.styler.write_comment(" !jvp");
        if (!has_vjp) try self.styler.write_comment(" !vjp");
    }
}

pub fn emit_region_start(self: *Self, depth: usize, region: pr.Region) !void {
    try self.writer.writeAll(self.indent);
    try self.writer.writeAll(self.indent);
    try self.emit_gutters(depth);
    try self.styler.write_region(self.styler.cfg.symbols.region_start);
    try self.styler.write_region(" ");
    try self.styler.write_region(region.name);
    try self.styler.write_region("[");
    try self.emit_annotations(region.annotations);
    try self.styler.write_region("]\n");
}

pub fn emit_region_end(self: *Self, depth: usize) !void {
    try self.writer.writeAll(self.indent);
    try self.writer.writeAll(self.indent);
    try self.emit_gutters(depth);
    try self.styler.write_region(self.styler.cfg.symbols.region_end);
    try self.styler.write_region("\n");
}

fn emit_annotations(self: *Self, annotations: []const pr.Annotation) !void {
    for (annotations, 0..) |annotation, index| {
        if (index > 0) try self.styler.write_region(", ");
        try self.styler.write_region(annotation.name);
        switch (annotation.value) {
            .unit => {},
            .boolean => |value| try self.writer.print("={}", .{value}),
            .integer => |value| try self.writer.print("={d}", .{value}),
            .floating_point => |value| try self.writer.print("={d}", .{value}),
            .string => |value| {
                try self.writer.writeAll("=\"");
                try std.zig.stringEscape(value, self.writer);
                try self.writer.writeAll("\"");
            },
            .strings => |values| {
                try self.writer.writeAll("=[");
                for (values, 0..) |value, value_index| {
                    if (value_index > 0) try self.writer.writeAll(", ");
                    try self.writer.writeAll("\"");
                    try std.zig.stringEscape(value, self.writer);
                    try self.writer.writeAll("\"");
                }
                try self.writer.writeAll("]");
            },
            .bytes => |value| try self.writer.print("=0x{x}", .{value}),
        }
    }
}

fn emit_gutters(self: *Self, count: usize) !void {
    var i: usize = 0;
    while (i < count) : (i += 1) {
        try self.styler.write_region(self.styler.cfg.symbols.region_gutter);
        try self.writer.writeAll(" ");
    }
}

fn emit_dtype_attr(self: *Self, op: *const pr.Op) !?bool {
    const prim = op.prim();
    const use_dtype = switch (prim) {
        .add, .subtract, .multiply, .divide, .maximum => true,
        .exp, .log, .rsqrt, .logistic, .convert => true,
        else => false,
    };
    if (!use_dtype) return null;
    if (op.inputs.len == 0) return false;
    const dtype = if (prim == .convert) blk: {
        break :blk op.params.convert;
    } else blk: {
        const tensor = op.operand(0).aval.as_tensor();
        break :blk tensor.dtype;
    };
    try self.writer.writeAll("dtype=");
    try self.styler.write_type_name(@tagName(dtype));
    return true;
}

fn is_dtype_only_attr(prim: pr.Prim) bool {
    return switch (prim) {
        .add, .subtract, .multiply, .divide, .maximum => true,
        .exp, .log, .rsqrt, .logistic, .convert => true,
        else => false,
    };
}
