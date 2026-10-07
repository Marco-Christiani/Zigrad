//! Registers op implementations and provides runtime dispatch.
//!
//! - `OpFor(prim)`: map from `pr.Prim` enum variant to its implementation struct.
//! - Runtime dispatch helpers (`validate`, `infer_output`, `vjp`, ...)
//!
//! ## Op interface
//!
//! Every op implementation is a struct with methods named per `op_methods`
//!  below. `validate` and `infer_output` are required. Missing them fails the
//!  build via the `validate_op_interface` comptime block. The others
//!  (`format`, `vjp`, `jvp`) are optional and probed
//!  via `@hasDecl` at dispatch time.
//!
//! Coverage: build with `-Demit-op-coverage=true` to have the registry
//!  `@compileLog` a per-op table of which interface methods are implemented.
//!  Used for tracking AD / lowering coverage without grepping the source.
//!
const std = @import("std");
const pr = @import("../pr.zig");
const build_options = @import("build_options");
const log = std.log.scoped(.@"zg/ops");

pub const types = @import("types.zig");
pub const constant = @import("constant.zig");
pub const elementwise = @import("elementwise.zig");
pub const unary = @import("unary.zig");
pub const compare = @import("compare.zig");
pub const contraction = @import("contraction.zig");
pub const shape = @import("shape.zig");
pub const special = @import("special.zig");

/// Map from Prim enum to op implementation struct.
pub fn OpFor(comptime prim: pr.Prim) type {
    return switch (prim) {
        .literal => constant.literal,
        .add => elementwise.add,
        .subtract => elementwise.subtract,
        .multiply => elementwise.multiply,
        .divide => elementwise.divide,
        .maximum => elementwise.maximum,
        .exp => unary.exp,
        .log => unary.log,
        .rsqrt => unary.rsqrt,
        .logistic => unary.logistic,
        .compare => compare.compare,
        .select => compare.select,
        .convert => unary.convert,
        .gather => shape.gather,
        .scatter => shape.scatter,
        .dot => contraction.dot,
        .mm => contraction.mm,
        .bmm => contraction.bmm,
        .dot_general => contraction.dot_general,
        .convolution => contraction.convolution,
        .reshape => shape.reshape,
        .iota => shape.iota,
        .transpose => shape.transpose,
        .broadcast_in_dim => shape.broadcast_in_dim,
        .slice => shape.slice,
        .concatenate => shape.concatenate,
        .reduce => shape.reduce,
        .call => special.call,
        .custom_call => special.custom_call,
    };
}

// Interface spec

/// Describes one method in the op interface.
///
/// A required method must exist on every op. Optional methods are probed via
///  `@hasDecl` at dispatch time.
const MethodSpec = struct {
    name: []const u8,
    required: bool,
    doc: []const u8,
};

/// The canonical op interface. Order matters only for the coverage report
///  layout.
const op_methods: []const MethodSpec = &.{
    .{ .name = "validate", .required = true, .doc = "Op construction sanity check" },
    // TODO(pr): Make inference optional for operations with explicit result types.
    .{ .name = "infer_output", .required = true, .doc = "Shape/dtype inference" },
    .{ .name = "format", .required = false, .doc = "IR dump formatting" },
    .{ .name = "vjp", .required = false, .doc = "Cotangent propagation for reverse-mode AD" },
    .{ .name = "jvp", .required = false, .doc = "Tangent propagation for forward-mode AD" },
};

/// Comptime enforcement of required interface methods.
///
/// Missing any required method is a hard build error identifying the
///  op and the missing method.
fn validate_op_interface() void {
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);
        inline for (op_methods) |m| {
            if (m.required and !@hasDecl(Op, m.name)) {
                @compileError("op " ++ @tagName(prim) ++ " missing required method: " ++ m.name);
            }
        }
    }
}

/// Comptime coverage report.
///
/// Emits one `@compileLog` per row so each line shows up as its own output
///  entry since a single multi-line `@compileLog` would serialize with `\n`
///  escaped, which is unreadable.
///
/// Opt-in via `-Demit-op-coverage=true`.
fn emit_op_coverage_report() void {
    @setEvalBranchQuota(100000);
    @compileLog("== op interface coverage ==");

    // Header row.
    comptime var header: []const u8 = std.fmt.comptimePrint("  {s: <18}", .{"op"});
    inline for (op_methods) |m| {
        header = header ++ std.fmt.comptimePrint(" {s: <13}", .{m.name});
    }
    @compileLog(header);

    // inline for unrolls, each op gets a row
    inline for (comptime std.enums.values(pr.Prim)) |prim| {
        const Op = OpFor(prim);
        comptime var row: []const u8 = std.fmt.comptimePrint("  {s: <18}", .{@tagName(prim)});
        inline for (op_methods) |m| {
            row = row ++ std.fmt.comptimePrint(" {s: <13}", .{
                if (@hasDecl(Op, m.name)) "yes" else "-",
            });
        }
        @compileLog(row);
    }

    @compileLog(@as([30]u8, @splat('-')) ++ "totals" ++ @as([30]u8, @splat('-')));
    inline for (op_methods) |m| {
        comptime var hits: usize = 0;
        inline for (comptime std.enums.values(pr.Prim)) |prim| {
            if (@hasDecl(OpFor(prim), m.name)) hits += 1;
        }
        const line = comptime std.fmt.comptimePrint("  {s: <13} {d}/{d}  {s}", .{
            m.name,
            hits,
            std.enums.values(pr.Prim).len,
            m.doc,
        });
        @compileLog(line);
    }
}

comptime {
    validate_op_interface();
    if (build_options.emit_op_coverage) emit_op_coverage_report();
}

// Dispatch Functions

/// Validate an op using the handler's validate function.
pub fn validate(op: *const pr.Op) pr.ValidationError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            return try OpFor(tag).validate(op, typed_params);
        },
    }
}

/// Infer output type for a set of params and inputs.
/// Called by FunctionBuilder before the Op exists.
pub fn infer_output(alloc: std.mem.Allocator, params: pr.Params, inputs: []const *pr.Var) pr.BuildError!pr.Aval {
    switch (params) {
        inline else => |typed_params, tag| {
            return try OpFor(tag).infer_output(alloc, inputs, typed_params);
        },
    }
}

/// Check if an op provides a local VJP rule.
pub fn has_local_vjp(prim: pr.Prim) bool {
    return switch (prim) {
        inline else => |p| @hasDecl(OpFor(p), "vjp"),
    };
}

/// Apply a local VJP rule to an operation.
pub fn vjp(ctx: types.VjpContext, op: *const pr.Op) types.AdError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "vjp")) {
                return try Handler.vjp(ctx, op, typed_params);
            }
            if (requires_vjp(ctx, op)) {
                if (!@import("builtin").is_test) log.err("{s} has no local VJP rule", .{@tagName(tag)});
                return error.MissingDerivativeRule;
            }
            return;
        },
    }
}

fn requires_vjp(ctx: types.VjpContext, op: *const pr.Op) bool {
    var active_output = false;
    for (op.outputs) |output| {
        if (types.is_differentiable(output.aval) and ctx.cotangent(output) != null) {
            active_output = true;
            break;
        }
    }
    if (!active_output) return false;

    for (op.inputs) |input| {
        if (types.is_differentiable(input.value.aval)) return true;
    }
    return false;
}

/// Check if an op provides a local JVP rule.
pub fn has_local_jvp(prim: pr.Prim) bool {
    return switch (prim) {
        inline else => |p| @hasDecl(OpFor(p), "jvp"),
    };
}

/// Apply a local JVP rule to an operation.
pub fn jvp(ctx: types.JvpContext, op: *const pr.Op) types.AdError!void {
    if (!requires_jvp(ctx, op)) return;

    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "jvp")) {
                return try Handler.jvp(ctx, op, typed_params);
            }
            if (!@import("builtin").is_test) log.warn("{s} has no local JVP rule", .{@tagName(tag)});
            return error.MissingDerivativeRule;
        },
    }
}

fn requires_jvp(ctx: types.JvpContext, op: *const pr.Op) bool {
    var has_differentiable_output = false;
    for (op.outputs) |output| {
        if (types.is_differentiable(output.aval)) {
            has_differentiable_output = true;
            break;
        }
    }
    if (!has_differentiable_output) return false;

    for (op.inputs) |input| {
        if (types.is_differentiable(input.value.aval) and
            ctx.tangent(input.value) != null) return true;
    }
    return false;
}

/// Format op-specific attributes.
pub fn format(writer: *types.Writer, op: *const pr.Op) types.FormatError!void {
    switch (op.params) {
        inline else => |typed_params, tag| {
            const Handler = OpFor(tag);
            if (@hasDecl(Handler, "format")) {
                return try Handler.format(writer, op, typed_params);
            }
            log.debug("{s} does not implement format()", .{@tagName(tag)});
            return;
        },
    }
}

test "local vjp support detection" {
    try std.testing.expect(has_local_vjp(.add));
    try std.testing.expect(has_local_vjp(.subtract));
    try std.testing.expect(has_local_vjp(.multiply));
    try std.testing.expect(has_local_vjp(.divide));
    try std.testing.expect(has_local_vjp(.dot));
    try std.testing.expect(has_local_vjp(.mm));
    try std.testing.expect(has_local_vjp(.bmm));
    try std.testing.expect(has_local_vjp(.reshape));
    try std.testing.expect(has_local_vjp(.transpose));
    try std.testing.expect(has_local_vjp(.broadcast_in_dim));
    try std.testing.expect(has_local_vjp(.reduce));
    try std.testing.expect(has_local_vjp(.exp));
    try std.testing.expect(has_local_vjp(.log));
    try std.testing.expect(has_local_vjp(.rsqrt));
    try std.testing.expect(has_local_vjp(.logistic));
    try std.testing.expect(has_local_vjp(.gather));
    try std.testing.expect(has_local_vjp(.select));
    try std.testing.expect(has_local_vjp(.dot_general));
    try std.testing.expect(has_local_vjp(.slice));
    try std.testing.expect(has_local_vjp(.concatenate));

    try std.testing.expect(!has_local_vjp(.literal));

    try std.testing.expect(has_local_vjp(.convert));

    try std.testing.expect(has_local_vjp(.maximum));
    try std.testing.expect(!has_local_vjp(.scatter));
    try std.testing.expect(!has_local_vjp(.compare));
    try std.testing.expect(!has_local_vjp(.call));
    try std.testing.expect(!has_local_vjp(.custom_call));
}

test "vjp rejects a missing rule on an active differentiable path" {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    var source_builder = try pr.FunctionBuilder.init(&program, "source");
    defer source_builder.deinit();
    const input = try source_builder.param_tensor(.f32, &.{4});
    const outputs = (try source_builder.custom_call(.{
        .target_name = "test.missing_vjp",
        .has_side_effect = false,
        .payload = &.{},
    }, &.{input}, &.{input.aval})).outputs;
    const source = try source_builder.finish(.{ .returns = outputs });

    var derived_builder = try pr.FunctionBuilder.init(&program, "derived");
    defer derived_builder.deinit();
    const cotangent = try derived_builder.param_tensor(.f32, &.{4});

    const primal_map = try std.testing.allocator.alloc(?*pr.Var, source.var_count);
    defer std.testing.allocator.free(primal_map);
    @memset(primal_map, null);
    const cotangent_map = try std.testing.allocator.alloc(?*pr.Var, source.var_count);
    defer std.testing.allocator.free(cotangent_map);
    @memset(cotangent_map, null);
    cotangent_map[outputs[0].id] = cotangent;

    const context = types.VjpContext{
        .builder = &derived_builder,
        .primals = primal_map,
        .cotangents = cotangent_map,
        .allocator = std.testing.allocator,
    };
    try std.testing.expectError(error.MissingDerivativeRule, vjp(context, source.ops[0]));

    cotangent_map[outputs[0].id] = null;
    try vjp(context, source.ops[0]);
}

test "local jvp support detection" {
    try std.testing.expect(has_local_jvp(.add));
    try std.testing.expect(has_local_jvp(.subtract));
    try std.testing.expect(has_local_jvp(.multiply));
    try std.testing.expect(has_local_jvp(.divide));

    try std.testing.expect(has_local_jvp(.exp));
    try std.testing.expect(has_local_jvp(.log));
    try std.testing.expect(has_local_jvp(.rsqrt));
    try std.testing.expect(has_local_jvp(.logistic));
    try std.testing.expect(has_local_jvp(.convert));

    try std.testing.expect(has_local_jvp(.reshape));
    try std.testing.expect(has_local_jvp(.transpose));
    try std.testing.expect(has_local_jvp(.broadcast_in_dim));
    try std.testing.expect(has_local_jvp(.reduce));
    try std.testing.expect(has_local_jvp(.slice));
    try std.testing.expect(has_local_jvp(.concatenate));
    try std.testing.expect(has_local_jvp(.gather));
    try std.testing.expect(!has_local_jvp(.iota));

    try std.testing.expect(has_local_jvp(.dot));
    try std.testing.expect(has_local_jvp(.mm));
    try std.testing.expect(has_local_jvp(.bmm));
    try std.testing.expect(has_local_jvp(.dot_general));

    try std.testing.expect(!has_local_jvp(.literal));

    try std.testing.expect(!has_local_jvp(.compare));
    try std.testing.expect(has_local_jvp(.select));

    try std.testing.expect(has_local_jvp(.maximum));
    try std.testing.expect(!has_local_jvp(.scatter));
    try std.testing.expect(!has_local_jvp(.call));
    try std.testing.expect(!has_local_jvp(.custom_call));
}
