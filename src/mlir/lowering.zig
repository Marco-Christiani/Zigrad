//! Shared PR to MLIR function scaffolding.
//!
//! Provides function construction, value mapping, type conversion, and
//!  serialization for a caller-supplied `LowerOpFn`.
//!
//! Explicit outline requests are resolved by PR passes before lowering.
//!
const std = @import("std");

const pr = @import("pr");
const mlir = @import("../c/mlir/mlir.zig");
const MlirSession = @import("session.zig").Session;

/// Serialized MLIR output encoding.
pub const OutputFormat = enum {
    mlir_text,
    mlir_bytecode,
};

const log = std.log.scoped(.@"zg/mlir_lower");

/// Maximum tensor rank representable by PR.
pub const max_rank = pr.max_rank;

/// Failures from selecting a program entry function.
pub const EntryError = error{EntryNotFound};

/// Failures exposed by PR to MLIR lowering.
///
/// `InvalidProgram` reports a missing value mapping or another structural
///  invariant that lowering requires. Callers validate PR before lowering.
pub const LowerError = mlir.Error || std.Io.Writer.Error || EntryError || error{InvalidProgram};

/// Callback type for dialect-specific op translation.
///
/// Each dialect module provides an implementation that maps a single PR op
///  to one or more MLIR ops in the target dialect, appending them to
///  `LowerContext.block`.
pub const LowerOpFn = *const fn (LowerContext, *const pr.Op) LowerError!void;

/// Per-function lowering state threaded through op lowering helpers.
///
/// `value_map` carries lowered SSA values from producers to consumers and uses
///  null for values that have not been lowered.
///
/// The caller provides arena-backed storage for this function lowering.
pub const LowerContext = struct {
    /// MLIR context for type/attribute construction.
    mlir_ctx: mlir.Context,
    /// The MLIR block being populated (function body).
    block: mlir.Block,
    /// Location attached to every generated op (currently file-level).
    loc: mlir.Location,
    /// `Var.id`-indexed map sized to `func.var_count`, with null for values not
    ///  yet lowered.
    value_map: []?mlir.Value,
    /// Scratch allocator for transient lowering buffers (e.g., dim arrays).
    arena: std.mem.Allocator,
    /// MLIR symbols keyed by PR function identity.
    function_symbols: *const std.AutoHashMapUnmanaged(pr.FunctionId, []const u8),

    pub fn get_value(self: LowerContext, v: *const pr.Var) ?mlir.Value {
        return self.value_map[v.id];
    }

    pub fn set_value(self: LowerContext, v: *const pr.Var, value: mlir.Value) void {
        self.value_map[v.id] = value;
    }

    pub fn tensor_to_mlir_type(self: LowerContext, t: pr.Tensor) mlir.Type {
        return tensor_type(self.mlir_ctx, t);
    }
};

/// Lower a PR program to MLIR bytecode or text.
///
/// The caller provides a configured session and the callback that translates
///  individual PR operations.
///
/// ## Entry function selection and naming
///
/// `entry` selects which PR function is the compilation entry point.
///
/// The selected function is emitted as `@main`. Other functions retain their
///  PR names unless that would collide with the entry symbol.
///
/// This operation does not run MLIR passes. Callers compose those operations
///  explicitly.
///
pub fn lower_program_to_mlir(
    allocator: std.mem.Allocator,
    session: MlirSession,
    program: *const pr.Program,
    entry: pr.FunctionId,
    out: OutputFormat,
    lower_op_fn: LowerOpFn,
) LowerError![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const ctx = session.ctx;
    const loc = mlir.Location.unknown(ctx);

    var module = mlir.Module.init(loc);
    defer module.deinit();

    if (program.get_function_by_id(entry) == null) return error.EntryNotFound;
    var function_symbols: std.AutoHashMapUnmanaged(pr.FunctionId, []const u8) = .empty;
    try function_symbols.ensureTotalCapacity(arena, @intCast(program.functions().len));
    for (program.function_ids()) |function_id| {
        const symbol = try choose_symbol_name(arena, program, function_id, entry);
        function_symbols.putAssumeCapacityNoClobber(function_id, symbol);
    }

    for (program.functions(), program.function_ids()) |func, function_id| {
        try lower_function_into_module(
            arena,
            ctx,
            module,
            func,
            function_symbols.get(function_id).?,
            &function_symbols,
            lower_op_fn,
        );
    }

    if (!module.op().verify()) return error.InvalidMlir;

    return try serialize_module(allocator, module, out);
}

/// Lower a single PR function into the MLIR module.
fn lower_function_into_module(
    arena: std.mem.Allocator,
    ctx: mlir.Context,
    module: mlir.Module,
    func: pr.Function,
    sym_name: []const u8,
    function_symbols: *const std.AutoHashMapUnmanaged(pr.FunctionId, []const u8),
    lower_op_fn: LowerOpFn,
) LowerError!void {
    const loc = mlir.Location.unknown(ctx);

    const param_types = try arena.alloc(mlir.Type, func.params.len);
    const param_locs = try arena.alloc(mlir.Location, func.params.len);
    for (func.params, 0..) |param_var, i| {
        const tensor = param_var.aval.as_tensor();
        param_types[i] = tensor_type(ctx, tensor);
        param_locs[i] = loc;
    }

    const result_types = try arena.alloc(mlir.Type, func.returns.len);
    for (func.returns, 0..) |ret_var, i| {
        const tensor = ret_var.aval.as_tensor();
        result_types[i] = tensor_type(ctx, tensor);
    }

    const fn_type = mlir.Type.function(ctx, param_types, result_types);

    const entry_block = try mlir.Block.init(param_types, param_locs);

    const value_map = try arena.alloc(?mlir.Value, func.var_count);
    @memset(value_map, null);
    for (func.params, 0..) |param_var, i| {
        value_map[param_var.id] = entry_block.argument(i);
    }

    const lower_ctx = LowerContext{
        .mlir_ctx = ctx,
        .block = entry_block,
        .loc = loc,
        .value_map = value_map,
        .arena = arena,
        .function_symbols = function_symbols,
    };

    for (func.ops) |op| try lower_op_fn(lower_ctx, op);

    const ret_values = try arena.alloc(mlir.Value, func.returns.len);
    for (func.returns, 0..) |ret_var, i| {
        ret_values[i] = value_map[ret_var.id] orelse return error.InvalidProgram;
    }

    const return_op = mlir.Operation.make(ctx, "func.return", .{
        .operands = ret_values,
        .verify = false,
        .location = loc,
    });
    entry_block.append_operation(return_op);

    const func_op = mlir.Operation.make(ctx, "func.func", .{
        .results = &.{},
        .blocks = &.{entry_block},
        .attributes = &.{
            .{ "sym_name", mlir.Attribute.string(ctx, sym_name) },
            .{ "function_type", mlir.Attribute.type_(fn_type) },
        },
        .verify = false,
        .location = loc,
    });
    module.get_body().append_operation(func_op);
}

/// Lower a `func.call` op. Uses the `func` dialect only.
pub fn lower_call(ctx: LowerContext, op: *const pr.Op) LowerError!void {
    const callee = ctx.function_symbols.get(op.params.call.callee) orelse
        return error.InvalidProgram;

    const operand_values = try ctx.arena.alloc(mlir.Value, op.inputs.len);
    for (op.inputs, 0..) |operand, i| {
        operand_values[i] = ctx.get_value(operand.value) orelse return error.InvalidProgram;
    }

    const result_types = try ctx.arena.alloc(mlir.Type, op.outputs.len);
    for (op.outputs, 0..) |out_var, i| {
        const out_tensor = out_var.aval.as_tensor();
        result_types[i] = ctx.tensor_to_mlir_type(out_tensor);
    }

    const callee_z = try ctx.arena.dupeZ(u8, callee);

    const mlir_op = mlir.Operation.make(ctx.mlir_ctx, "func.call", .{
        .results = result_types,
        .operands = operand_values,
        .attributes = &.{
            .{ "callee", mlir.Attribute.symbol(ctx.mlir_ctx, callee_z) },
        },
        .verify = false,
        .location = ctx.loc,
    });

    ctx.block.append_operation(mlir_op);
    for (op.outputs, 0..) |out_var, i| {
        ctx.set_value(out_var, mlir_op.result(i));
    }
}

/// Convert a PR element type to its builtin MLIR scalar type.
pub fn dtype_to_mlir_type(ctx: mlir.Context, dt: pr.DType) mlir.Type {
    return switch (dt) {
        .f16 => mlir.Type.float(ctx, .f16),
        .bf16 => mlir.Type.float(ctx, .bf16),
        .f32 => mlir.Type.float(ctx, .f32),
        .f64 => mlir.Type.float(ctx, .f64),
        .i8 => mlir.Type.int(ctx, .i8),
        .u8 => mlir.Type.int(ctx, .i8),
        .i32 => mlir.Type.int(ctx, .i32),
        .i64 => mlir.Type.int(ctx, .i64),
        .u32 => mlir.Type.int(ctx, .i32),
        .u64 => mlir.Type.int(ctx, .i64),
        .bool => mlir.Type.int(ctx, .i1),
    };
}

/// Convert a PR tensor type to a ranked MLIR tensor type.
pub fn tensor_type(ctx: mlir.Context, t: pr.Tensor) mlir.Type {
    var buf: [max_rank]i64 = undefined;
    const dims_i64 = buf[0..t.shape.dims.len];
    for (t.shape.dims, 0..) |d, i| dims_i64[i] = @intCast(d);
    return mlir.Type.tensor(dims_i64, dtype_to_mlir_type(ctx, t.dtype));
}

const SerializeError = std.mem.Allocator.Error || std.Io.Writer.Error;

fn serialize_module(allocator: std.mem.Allocator, module: mlir.Module, out: OutputFormat) SerializeError![]u8 {
    var writer_state = std.Io.Writer.Allocating.init(allocator);
    defer writer_state.deinit();

    switch (out) {
        .mlir_bytecode => try module.op().write_bytecode(&writer_state.writer),
        .mlir_text => try module.op().print(&writer_state.writer, .{}),
    }

    return try writer_state.toOwnedSlice();
}

/// Choose an MLIR symbol without colliding with the emitted `@main` entry.
fn choose_symbol_name(
    arena: std.mem.Allocator,
    program: *const pr.Program,
    function_id: pr.FunctionId,
    entry: pr.FunctionId,
) error{OutOfMemory}![]const u8 {
    if (function_id == entry) return "main";

    const func = program.get_function_by_id(function_id).?;
    if (!std.mem.eql(u8, func.name, "main")) return func.name;

    var suffix: usize = 0;
    while (true) : (suffix += 1) {
        const candidate = try std.fmt.allocPrint(arena, "main_non_entry_{d}", .{suffix});
        if (!is_symbol_name_used(program, entry, candidate)) {
            if (!@import("builtin").is_test) log.warn("renaming non-entry function 'main' to '{s}' to avoid entry collision", .{candidate});
            return candidate;
        }
    }
}

fn is_symbol_name_used(program: *const pr.Program, entry: pr.FunctionId, name: []const u8) bool {
    for (program.functions(), program.function_ids()) |func, function_id| {
        if (function_id == entry) continue;
        if (std.mem.eql(u8, func.name, name)) return true;
    }
    return false;
}
