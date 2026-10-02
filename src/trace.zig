//! User-facing tracing and trace-time program construction.
//!
//! ## Typical usage
//!
//! Compose AD and an optimizer inside one traced program:
//! ```zig
//! fn train_step(params: Params, batch: Batch) !struct { loss: Tensor, updated: Params } {
//!     var vg = try zg.transforms.value_and_grad(loss_fn, .{ params, batch }, .{});
//!     defer vg.deinit();
//!     // Apply an optimizer to `vg.grads` and return the updated parameters.
//! }
//! var traced = try zg.trace(
//!     train_step,
//!     allocator,
//!     specs,
//!     .{ .name = "train_step" },
//! );
//! defer traced.deinit();
//! ```
const std = @import("std");

const callable = @import("callable.zig");
const pr = @import("pr");
const Tensor = @import("tensor.zig");
const TensorTree = @import("utils.zig").Tree(Tensor);

/// Options for tracing a function.
pub const Options = struct {
    /// Name assigned to the traced PR function.
    name: []const u8 = "main",
};

/// Trace a comptime function and retain its call contract.
///
/// Each tensor leaf contributes its data type and shape. Abstract, host, and
///  device tensors are valid because tracing does not read their data.
///
/// Each leaf becomes a traced parameter. Struct and tuple nesting are preserved,
///  and the function receives traced tensors in the same structure. The caller
///  remains responsible for `specs`.
///
/// The traced function may compose AD transforms. Its return value defines the
///  program outputs.
///
/// The result owns its PR program. `deinit` releases that program, and `bind`
///  accepts a loaded program compiled from `Traced.program`.
pub fn trace(
    comptime func: anytype,
    allocator: std.mem.Allocator,
    specs: anytype,
    comptime opts: Options,
) !callable.Traced(func, @TypeOf(specs)) {
    var program = pr.Program.init(allocator);
    errdefer program.deinit();
    const function = try trace_into(func, allocator, &program, specs, opts.name);
    try program.set_entry(function);
    return .{ .program = program, .function = function };
}

/// Trace a comptime function into an existing PR program.
///
/// The returned identity addresses the registered function for the lifetime of
///  `program`. Functions generated during tracing are registered in the same
///  program.
///
///  Restores to progam checkpoint on failure, see `pr.Program.restore_appends`.
pub fn trace_into(
    comptime func: anytype,
    /// Allocator for temporary tree and value-list storage.
    scratch: std.mem.Allocator,
    /// Program that owns the traced function and anything it generates.
    program: *pr.Program,
    specs: anytype,
    /// Unique label assigned to the registered function.
    name: []const u8,
) !pr.FunctionId {
    const SpecType = @TypeOf(specs);

    const saved = program.checkpoint_appends();
    errdefer program.restore_appends(saved);

    const spec_tree = try TensorTree.from(program.allocator(), specs);

    var builder = try pr.FunctionBuilder.init(program, name);
    defer builder.deinit();

    const traced = try spec_tree.map(Tensor, &builder, struct {
        fn f(b: *pr.FunctionBuilder, leaf: Tensor) !Tensor {
            return try Tensor.param(b, leaf.dtype, leaf.shape.const_slice());
        }
    }.f);

    const structured = try traced.extract(SpecType);
    const args = if (@typeInfo(SpecType) == .@"struct" and @typeInfo(SpecType).@"struct".is_tuple)
        structured
    else
        .{structured};
    const result_raw = if (comptime is_generated_callable(@TypeOf(func)))
        @TypeOf(func).call(args)
    else
        @call(.auto, func, args);
    const result = switch (@typeInfo(@TypeOf(result_raw))) {
        .error_union => try result_raw,
        else => result_raw,
    };

    const output_tensors = try TensorTree.flatten(scratch, result);
    defer scratch.free(output_tensors);
    if (output_tensors.len == 0) return error.NoOutputs;

    const output_vars = try scratch.alloc(*pr.Var, output_tensors.len);
    defer scratch.free(output_vars);
    for (output_tensors, 0..) |t, i| output_vars[i] = try t.get_var();

    const func_pr = try builder.finish(.{ .returns = output_vars });
    return try program.add_function(func_pr);
}

fn is_generated_callable(comptime T: type) bool {
    return switch (@typeInfo(T)) {
        .@"struct" => @hasDecl(T, "ArgsType") and
            @hasDecl(T, "ResultType") and
            @hasDecl(T, "call"),
        else => false,
    };
}

fn trace_test_identity(input: Tensor) Tensor {
    return input;
}

fn trace_test_double(input: Tensor) !Tensor {
    return try input.add(input);
}

test trace_into {
    var program = pr.Program.init(std.testing.allocator);
    defer program.deinit();

    const specs = .{Tensor.abstract(.f32, &.{2})};
    const identity = try trace_into(
        trace_test_identity,
        std.testing.allocator,
        &program,
        specs,
        "identity",
    );
    const double = try trace_into(
        trace_test_double,
        std.testing.allocator,
        &program,
        specs,
        "double",
    );

    try std.testing.expect(identity != double);
    try std.testing.expectEqual(@as(usize, 2), program.functions().len);
    try std.testing.expectEqualStrings("identity", program.get_function_by_id(identity).?.name);
    try std.testing.expectEqualStrings("double", program.get_function_by_id(double).?.name);
    try std.testing.expectEqual(@as(?pr.FunctionId, null), program.entry);
    try pr.validate_program(&program);
}

test trace {
    const specs = .{Tensor.abstract(.f32, &.{2})};
    var traced = try trace(
        trace_test_double,
        std.testing.allocator,
        specs,
        .{ .name = "double" },
    );
    defer traced.deinit();

    try std.testing.expectEqual(traced.function, try traced.program.resolve_entry());
}
