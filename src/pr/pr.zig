//! Toolchain-neutral program representation.
//!
//! A `Program` arena stores functions, operations, SSA values, operands, and
//!  parameter slices. `Program.deinit` releases the arena in one operation.
//! Lowering remains an explicit compilation operation outside PR.
//!
//! ## Design invariants
//!
//! 1. Operations have one result except `call` and `custom_call`.
//! 2. `FunctionBuilder.finish` validates through the operation registry.
//! 3. Builder methods copy slice parameters into the program arena.
const std = @import("std");
const ops = @import("ops/ops.zig");
const pr_log = std.log.scoped(.@"zg/pr");
const Allocator = std.mem.Allocator;

pub const DType = @import("dtype").DType;

/// Shape represented by a dimension slice.
pub const Shape = struct {
    dims: []const i64,

    pub fn rank(self: Shape) usize {
        return self.dims.len;
    }

    pub fn num_elements(self: Shape) usize {
        var count: usize = 1;
        for (self.dims) |d| count *= @intCast(d);
        return count;
    }

    fn dupe(self: Shape, allocator: Allocator) Allocator.Error!Shape {
        return .{ .dims = try allocator.dupe(i64, self.dims) };
    }
};

/// Maximum rank stored by `BoundedShape`.
pub const max_rank = 8;

/// Stack-allocated shape with a fixed maximum rank.
pub const BoundedShape = struct {
    /// TODO(shape): Define the dimension scalar and host-index conversion policy.
    buf: [max_rank]i64 = undefined,

    // TODO(shape): Reevaluate the separate length field against a slice-backed
    //  representation before changing fixed-capacity shape storage.
    len: usize = 0,

    pub fn from_slice(s: []const i64) BoundedShape {
        var result = BoundedShape{};
        for (s) |d| result.append_assume_capacity(d);
        return result;
    }

    pub fn const_slice(self: *const BoundedShape) []const i64 {
        return self.buf[0..self.len];
    }

    pub fn append_assume_capacity(self: *BoundedShape, val: i64) void {
        self.buf[self.len] = val;
        self.len += 1;
    }

    pub fn rank(self: BoundedShape) usize {
        return self.len;
    }

    pub fn num_elements(self: BoundedShape) usize {
        var count: usize = 1;
        for (self.const_slice()) |d| count *= @intCast(d);
        return count;
    }

    /// Format shape for display.
    pub fn format(shape: BoundedShape, allocator: std.mem.Allocator) ![]const u8 {
        const dims = shape.const_slice();
        if (dims.len == 0) return try allocator.dupe(u8, "scalar");

        var result = std.ArrayList(u8).initCapacity(allocator, 32) catch
            return try allocator.dupe(u8, "[...]");
        defer result.deinit(allocator);

        const writer = result.writer(allocator);
        try writer.writeAll("[");
        for (dims, 0..) |d, i| {
            if (i > 0) try writer.writeAll(", ");
            try writer.print("{d}", .{d});
        }
        try writer.writeAll("]");

        return try result.toOwnedSlice(allocator);
    }
};

/// Abstract value: the type of a Var without its data.
///
/// ## ADR
/// The union leaves room for non-tensor values without changing the `Var`
///  layout.
pub const Aval = union(enum) {
    tensor: Tensor,

    /// Copy this abstract value and its nested storage with `allocator`.
    pub fn dupe(self: Aval, allocator: Allocator) Allocator.Error!Aval {
        return switch (self) {
            inline else => |s, t| @unionInit(Aval, @tagName(t), try s.dupe(allocator)),
        };
    }

    /// Extract the tensor type. Exhaustive over Aval variants.
    pub fn as_tensor(self: Aval) Tensor {
        return switch (self) {
            .tensor => |t| t,
        };
    }
};

/// PR tensor descriptor containing a data type and shape.
///
/// It carries no data, device, or buffer.
/// TODO(pr): Rename this descriptor and its `Aval` tag to distinguish them
///  from runtime tensors.
pub const Tensor = struct {
    dtype: DType,
    shape: Shape,

    fn dupe(self: Tensor, allocator: Allocator) Allocator.Error!Tensor {
        return .{
            .dtype = self.dtype,
            .shape = try self.shape.dupe(allocator),
        };
    }
};

/// Scalar constant keyed by its `DType`.
/// Float16 variants store the raw u16 bit pattern.
pub const Literal = union(DType) {
    f16: u16,
    bf16: u16,
    f32: f32,
    f64: f64,
    i8: i8,
    u8: u8,
    i32: i32,
    i64: i64,
    u32: u32,
    u64: u64,
    bool: bool,

    pub fn dtype(self: Literal) DType {
        return std.meta.activeTag(self);
    }

    /// Create a scalar literal by converting from f64.
    pub fn from_f64(value_dtype: DType, value: f64) Literal {
        return switch (value_dtype) {
            .f16 => .{ .f16 = DType.f16.encode(f64, value) },
            .bf16 => .{ .bf16 = DType.bf16.encode(f64, value) },
            .f32 => .{ .f32 = @floatCast(value) },
            .f64 => .{ .f64 = value },
            .i8 => .{ .i8 = @intFromFloat(value) },
            .u8 => .{ .u8 = @intFromFloat(value) },
            .i32 => .{ .i32 = @intFromFloat(value) },
            .i64 => .{ .i64 = @intFromFloat(value) },
            .u32 => .{ .u32 = @intFromFloat(value) },
            .u64 => .{ .u64 = @intFromFloat(value) },
            // TODO(pr): Define boolean conversion semantics for NaN.
            .bool => .{ .bool = value != 0.0 },
        };
    }
};

/// Primitive operation tag.
///
/// Each variant corresponds to a Params payload type and an entry in the op
///  registry (ops/ops.zig) providing at least `validate` and `infer_output`.
pub const Prim = enum {
    literal,
    add,
    subtract,
    multiply,
    divide,
    maximum,
    exp,
    log,
    rsqrt,
    logistic,
    compare,
    select,
    convert,
    gather,
    scatter,
    dot,
    mm,
    bmm,
    dot_general,
    convolution,
    reshape,
    iota,
    broadcast_in_dim,
    transpose,
    slice,
    concatenate,
    reduce,
    call,
    custom_call,
};

/// Parameters for gather (indexed read from a tensor).
///
/// Given an operand and an indices tensor, extracts slices from the operand at
///  positions given by indices.
pub const GatherParams = struct {
    /// Size of the slice extracted along each operand dimension.
    slice_sizes: []const i64,
    /// Dimensions of the output that correspond to the slice window.
    offset_dims: []const i64,
    /// Slice dimensions that are collapsed (size must be 1).
    collapsed_slice_dims: []const i64,
    /// Maps each index vector element to an operand dimension.
    start_index_map: []const i64,
    /// Which dimension of the indices tensor holds the index vector.
    index_vector_dim: i64,
};

pub const CompareParams = struct {
    direction: CompareDirection,
    compare_type: CompareType,
};

pub const CompareDirection = enum {
    EQ,
    NE,
    GE,
    GT,
    LE,
    LT,
};

pub const CompareType = enum {
    SIGNED,
    UNSIGNED,
    FLOAT,
    TOTALORDER,
};

/// Parameters for scatter (indexed write into a tensor).
///
/// Writes `updates` into `operand` at positions given by `indices`, combining
///  with `reduction`.
pub const ScatterParams = struct {
    /// Dimensions of `updates` that correspond to the update window.
    update_window_dims: []const i64,
    /// Window dimensions that are inserted (size 1) rather than present in updates.
    inserted_window_dims: []const i64,
    /// Maps each index vector element to an operand dimension.
    scatter_dims_to_operand_dims: []const i64,
    /// Which dimension of the indices tensor holds the index vector.
    index_vector_dim: i64,
    /// How overlapping updates are combined.
    reduction: ScatterReduction = .add,
};

pub const ScatterReduction = enum {
    add,
    max,
    min,
    mul,
};

/// Parameters for slice (contiguous sub-tensor extraction).
pub const SliceParams = struct {
    /// Per-dimension inclusive start offsets.
    start_indices: []const i64,
    /// Per-dimension exclusive end offsets.
    limit_indices: []const i64,
    /// Per-dimension step sizes (1 = dense).
    strides: []const i64,
};

/// Parameters for generalized dot product.
///
/// Contracting dims are reduced.
/// Batch dims are preserved as leading output dimensions.
/// Remaining dims become the non-contracted output dimensions.
pub const DotGeneralParams = struct {
    lhs_batch_dims: []const i64,
    rhs_batch_dims: []const i64,
    lhs_contracting_dims: []const i64,
    rhs_contracting_dims: []const i64,
};

/// Maps the logical batch, feature, and spatial axes onto tensor dimensions.
pub const ConvolutionDimensionNumbers = struct {
    input_batch_dimension: i64,
    input_feature_dimension: i64,
    input_spatial_dimensions: []const i64,
    kernel_input_feature_dimension: i64,
    kernel_output_feature_dimension: i64,
    kernel_spatial_dimensions: []const i64,
    output_batch_dimension: i64,
    output_feature_dimension: i64,
    output_spatial_dimensions: []const i64,
};

/// Defines the convolution window, axis mapping, and grouping semantics.
pub const ConvolutionParams = struct {
    /// Step between adjacent windows along each spatial axis.
    window_strides: []const i64,
    /// Low and high padding flattened as `[low0, high0, low1, high1, ...]`.
    padding: []const i64,
    /// Spacing between input elements along each spatial axis.
    lhs_dilation: []const i64,
    /// Spacing between kernel elements along each spatial axis.
    rhs_dilation: []const i64,
    /// Reverse each selected spatial window before contraction.
    window_reversal: []const bool,
    /// Physical axis mappings for the input, kernel, and output tensors.
    dimensions: ConvolutionDimensionNumbers,
    /// Independent groups partitioning the input and output features.
    feature_group_count: i64 = 1,
    /// Independent groups partitioning the input batch and kernel outputs.
    batch_group_count: i64 = 1,
};

pub const ReshapeParams = struct {
    out_shape: []const i64,
};

/// Parameters for generating a tensor of incrementing indices.
pub const IotaParams = struct {
    out_shape: []const i64,
    out_dtype: DType,
    /// Which dimension varies (0-indexed). E.g. for shape [2,3] with
    /// dimension=1, produces [[0,1,2],[0,1,2]].
    dimension: i64,
};

/// Parameters for (symbolically) expanding a tensor to a larger shape.
pub const BroadcastInDimParams = struct {
    /// Target shape of the output tensor.
    out_shape: []const i64,
    /// Maps each input dimension to its position in the output shape.
    /// Length must equal the input rank. E.g. for a [3] input broadcast
    /// to [2,3], dimensions = &.{1} means input dim 0 maps to output dim 1.
    dimensions: []const i64,
};

pub const TransposeParams = struct {
    permutation: []const i64,
};

pub const ConcatenateParams = struct {
    axis: i64,
};

pub const Reduction = enum {
    sum,
    maximum,
};

pub const ReduceParams = struct {
    axes: []const i64,
    operation: Reduction,
};

/// Identity of a registered function within a `Program`.
///
/// Appending or replacing functions preserves existing identities. An identity
///  assigned to a removed function is not reused by the same program.
pub const FunctionId = enum(u32) { _ };

/// Failures from registering a function with a program.
pub const FunctionRegistrationError = Allocator.Error || error{
    DuplicateFunctionName,
    DuplicateFunctionId,
    FunctionIdExhausted,
};

const FunctionEntry = struct {
    id: FunctionId,
    function: Function,
};

pub const CallParams = struct {
    callee: FunctionId,
};

/// Parameters for an opaque runtime-dispatched custom call operation.
pub const CustomCallParams = struct {
    /// Dispatch identity interpreted by the selected execution integration.
    target_name: []const u8,
    /// Whether the call has observable effects not represented by its results.
    has_side_effect: bool,
    /// Target-defined semantic data embedded in the operation.
    payload: []const u8 = &.{},
};

/// Parameter union keyed by `Prim`.
pub const Params = union(Prim) {
    literal: Literal,
    add: void,
    subtract: void,
    multiply: void,
    divide: void,
    maximum: void,
    exp: void,
    log: void,
    rsqrt: void,
    logistic: void,
    compare: CompareParams,
    select: void,
    convert: DType,
    gather: GatherParams,
    scatter: ScatterParams,
    dot: void,
    mm: void,
    bmm: void,
    dot_general: DotGeneralParams,
    convolution: ConvolutionParams,
    reshape: ReshapeParams,
    iota: IotaParams,
    broadcast_in_dim: BroadcastInDimParams,
    transpose: TransposeParams,
    slice: SliceParams,
    concatenate: ConcatenateParams,
    reduce: ReduceParams,
    call: CallParams,
    custom_call: CustomCallParams,

    fn dupe(self: Params, allocator: Allocator) Allocator.Error!Params {
        return switch (self) {
            inline else => |value, tag| @unionInit(
                Params,
                @tagName(tag),
                try dupe_param_value(@TypeOf(value), allocator, value),
            ),
        };
    }
};

fn dupe_param_value(comptime T: type, allocator: Allocator, value: T) Allocator.Error!T {
    return switch (@typeInfo(T)) {
        .void, .bool, .int, .float, .@"enum" => value,
        .optional => |info| if (value) |payload|
            try dupe_param_value(info.child, allocator, payload)
        else
            null,
        .pointer => |info| blk: {
            if (info.size != .slice)
                @compileError("slices are the only supported pointer parameter type");
            const result = try allocator.alloc(info.child, value.len);
            for (value, result) |item, *stored|
                stored.* = try dupe_param_value(info.child, allocator, item);
            break :blk result;
        },
        .@"struct" => |info| blk: {
            var result: T = undefined;
            inline for (info.field_names, info.field_types) |field_name, field_type|
                @field(result, field_name) = try dupe_param_value(field_type, allocator, @field(value, field_name));
            break :blk result;
        },
        .@"union" => |info| blk: {
            const Tag = info.tag_type orelse
                @compileError("PR parameters require tagged unions");
            const tag = std.meta.activeTag(value);
            inline for (info.field_names, info.field_types) |field_name, field_type| {
                if (tag == @field(Tag, field_name)) {
                    break :blk @unionInit(
                        T,
                        field_name,
                        try dupe_param_value(field_type, allocator, @field(value, field_name)),
                    );
                }
            }
            unreachable;
        },
        else => @compileError("unsupported PR parameter type: " ++ @typeName(T)),
    };
}

/// SSA value with inline type and intrinsic use-list
///
/// Analogous to mlir::Value.
pub const Var = struct {
    id: u32,
    aval: Aval,
    /// The op that produces this var, or null for function parameters.
    defining_op: ?*const Op = null,
    /// Head of doubly-linked use-list.
    first_use: ?*Operand = null,

    pub fn as_tensor(self: *const Var) Tensor {
        return self.aval.as_tensor();
    }

    pub fn is_unused(self: *const Var) bool {
        return self.first_use == null;
    }

    pub fn has_one_use(self: *const Var) bool {
        const first = self.first_use orelse return false;
        return first.next == null;
    }

    /// Returns the sole consuming operation of this value.
    ///
    /// Function returns do not add nodes to the intrusive use list. Returns
    ///  null when this value has zero or several consuming operations. The
    ///  returned pointer shares this value's program lifetime.
    pub fn only_user(self: *const Var) ?*const Op {
        const first = self.first_use orelse return null;
        if (first.next != null) return null;
        return first.owner;
    }

    /// Checks if var has *at least* n uses.
    pub fn has_n_uses(self: *const Var, n: u32) bool {
        var count: u32 = 0;
        var cur = self.first_use;
        while (cur) |use| {
            count += 1;
            if (count >= n) return true;
            cur = use.next;
        }
        return false;
    }

    /// Replace all uses of this var with `new`. Transfers the use-list.
    pub fn replace_all_uses_with(self: *Var, new: *Var) void {
        if (self == new) return;
        while (self.first_use) |use| use.set(new);
    }
};

/// A use-chain node linking a consuming Op to the Var it reads.
pub const Operand = struct {
    value: *Var,
    owner: *const Op,
    index: u32,
    prev: ?*Operand = null,
    next: ?*Operand = null,

    /// Initialize this stable operand slot and attach it to `value`'s use-list.
    pub fn attach(self: *Operand, value: *Var, owner: *const Op, index: u32) void {
        std.debug.assert(index < owner.inputs.len);
        std.debug.assert(&owner.inputs[index] == self);
        self.* = .{
            .value = value,
            .owner = owner,
            .index = index,
        };
        self.link(value);
    }

    /// Rebind this use-site to `new`, preserving def-use invariants.
    ///
    /// This moves one `Operand` node from the old value's use-list to the head
    ///  of `new`'s use-list. List order maintains links and does not encode
    ///  program execution order.
    ///
    /// Before:
    /// ```
    /// old.first_use -> A <-> self <-> C
    /// new.first_use -> N1 <-> N2
    /// ```
    ///
    /// After:
    /// ```
    /// old.first_use -> A <-> C
    /// new.first_use -> self <-> N1 <-> N2
    /// ```
    ///
    /// When `self` is the old head, `old.first_use` becomes `self.next`.
    ///
    /// ## ADR
    ///
    /// LLVM commonly stores `prev` as `?*?*Operand`. The direct previous-node
    ///  pointer used here keeps unlinking explicit.
    pub fn set(self: *Operand, new: *Var) void {
        const old = self.value;
        if (old == new) return;

        self.unlink(old);
        self.value = new;
        self.link(new);
    }

    fn unlink(self: *Operand, value: *Var) void {
        if (self.prev) |p| {
            p.next = self.next;
        } else {
            value.first_use = self.next;
        }
        if (self.next) |n| n.prev = self.prev;
    }

    fn link(self: *Operand, value: *Var) void {
        self.prev = null;
        self.next = value.first_use;
        if (value.first_use) |head| head.prev = self;
        value.first_use = self;
    }
};

/// An operation in the IR.
pub const Op = struct {
    id: u32,
    inputs: []Operand,
    outputs: []*Var,
    params: Params,

    pub fn prim(self: *const Op) Prim {
        return std.meta.activeTag(self.params);
    }

    pub fn operand(self: *const Op, i: usize) *Var {
        return self.inputs[i].value;
    }

    pub fn result(self: *const Op, i: usize) *Var {
        return self.outputs[i];
    }

    /// True if all results are unused.
    pub fn use_empty(self: *const Op) bool {
        for (self.outputs) |out| {
            if (!out.is_unused()) return false;
        }
        return true;
    }
};

/// Value carried by a named IR annotation.
pub const AnnotationValue = union(enum) {
    unit,
    boolean: bool,
    integer: i64,
    floating_point: f64,
    string: []const u8,
    strings: []const []const u8,
    bytes: []const u8,

    /// Return the string payload or null for another storage kind.
    pub fn as_string(self: AnnotationValue) ?[]const u8 {
        return switch (self) {
            .string => |value| value,
            else => null,
        };
    }
};

/// A namespaced compiler annotation.
///
/// Annotation names define their own value contracts. PR stores unknown names
///  without requiring registration in a central registry.
pub const Annotation = struct {
    /// Namespaced contract name interpreted by an owning pass.
    name: []const u8,
    /// Contract payload retained by PR without interpreting its meaning.
    value: AnnotationValue,
};

/// A named group of PR ops with attached steering metadata.
/// Materialized at `FunctionBuilder.finish()` from the annotation stack by default.
pub const Region = struct {
    id: u32,
    name: []const u8,
    /// Compiler metadata attached to this operation group.
    annotations: []const Annotation,
    /// Stable ids of member ops, in program order.
    op_ids: []const u32,

    /// Find an annotation by its exact namespaced name.
    pub fn find_annotation(self: Region, name: []const u8) ?*const Annotation {
        for (self.annotations) |*annotation| {
            if (std.mem.eql(u8, annotation.name, name)) return annotation;
        }
        return null;
    }
};

/// A named function in the program: parameter vars, a linear op sequence,
///  return vars, and optional annotation regions.
///
/// Constructed by `FunctionBuilder.finish()`, which validates all ops.
/// Rewrites may mutate values and operands after construction.
/// TODO(pr): Define and enforce the permitted mutation surface after
///  `FunctionBuilder.finish()`.
pub const Function = struct {
    name: []const u8,
    /// Compiler metadata attached to this callable unit.
    annotations: []const Annotation = &.{},
    params: []*Var,
    returns: []*Var,
    ops: []*Op,
    regions: []const Region,
    var_count: u32,

    /// Find an annotation by its exact namespaced name.
    pub fn find_annotation(self: Function, name: []const u8) ?*const Annotation {
        for (self.annotations) |*annotation| {
            if (std.mem.eql(u8, annotation.name, name)) return annotation;
        }
        return null;
    }

    /// Return regions that satisfy a predicate.
    pub fn regions_matching(self: Function, predicate: *const fn (Region) bool) RegionIterator {
        return .{ .regions = self.regions, .predicate = predicate, .index = 0 };
    }

    pub fn op_by_id(self: Function, id: u32) ?*Op {
        for (self.ops) |op| {
            if (op.id == id) return op;
        }
        return null;
    }

    pub fn op_index_by_id(self: Function, id: u32) ?usize {
        for (self.ops, 0..) |op, index| {
            if (op.id == id) return index;
        }
        return null;
    }
};

pub const RegionIterator = struct {
    regions: []const Region,
    predicate: *const fn (Region) bool,
    index: usize,

    pub fn next(self: *RegionIterator) ?Region {
        while (self.index < self.regions.len) {
            const region = self.regions[self.index];
            self.index += 1;
            if (self.predicate(region)) return region;
        }
        return null;
    }
};

/// Top-level container for a PR program.
///
/// Owns an arena allocator that backs all IR nodes (Vars, Ops, Operands,
///  param slices, Functions). Call `deinit` to free everything in bulk.
///  Individual nodes are never freed separately.
///
/// A program may select one registered function as its compilation entry.
/// `resolve_entry` infers the entry only when the call graph has one root.
pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    function_entries: std.MultiArrayList(FunctionEntry),
    function_index_by_id: std.AutoHashMapUnmanaged(FunctionId, usize),
    assigned_function_ids: std.AutoHashMapUnmanaged(FunctionId, void),
    function_id_by_name: std.StringHashMapUnmanaged(FunctionId),
    next_function_id: u64,
    reserved_function_names: std.ArrayList([]const u8),
    /// Explicit compilation entry. A null value requests call-graph inference.
    entry: ?FunctionId,

    /// Errors reported when selecting a program entry function.
    pub const EntryResolutionError = error{
        NoFunctions,
        NoRootFunction,
        EntrySelectionRequired,
        EntryFunctionMissing,
    };

    /// Program state restored when discarding appended construction work.
    pub const AppendCheckpoint = struct {
        function_count: usize,
        reservation_count: usize,
        entry: ?FunctionId,
    };

    pub fn init(backing_allocator: std.mem.Allocator) Program {
        return .{
            .arena = std.heap.ArenaAllocator.init(backing_allocator),
            .function_entries = .empty,
            .function_index_by_id = .empty,
            .assigned_function_ids = .empty,
            .function_id_by_name = .empty,
            .next_function_id = 0,
            .reserved_function_names = .empty,
            .entry = null,
        };
    }

    pub fn allocator(self: *Program) std.mem.Allocator {
        return self.arena.allocator();
    }

    /// Register a function under the next program-local identity.
    pub fn add_function(
        self: *Program,
        /// Function whose referenced storage remains valid for the program lifetime.
        func: Function,
    ) FunctionRegistrationError!FunctionId {
        const id: FunctionId = @fromBackingInt(@intCast(std.math.cast(u32, self.next_function_id) orelse
            return error.FunctionIdExhausted));
        return try self.add_function_with_id(func, id);
    }

    /// Register a function under an unused program-local identity.
    ///
    /// The next allocated identity advances past `id`. This preserves source
    ///  identities when reconstructing or projecting a program.
    pub fn add_function_with_id(
        self: *Program,
        /// Function whose referenced storage remains valid for the program lifetime.
        func: Function,
        /// Identity to assign to `func`.
        id: FunctionId,
    ) FunctionRegistrationError!FunctionId {
        if (self.function_id_by_name.contains(func.name)) return error.DuplicateFunctionName;
        if (self.assigned_function_ids.contains(id)) return error.DuplicateFunctionId;
        const index = self.function_entries.len;
        const arena = self.allocator();

        try self.function_entries.append(arena, .{ .id = id, .function = func });
        errdefer self.function_entries.shrinkRetainingCapacity(index);
        try self.function_index_by_id.put(arena, id, index);
        errdefer std.debug.assert(self.function_index_by_id.remove(id));
        try self.function_id_by_name.put(arena, func.name, id);
        errdefer std.debug.assert(self.function_id_by_name.remove(func.name));
        try self.assigned_function_ids.put(arena, id, {});

        self.next_function_id = @max(self.next_function_id, @as(u64, @backingInt(id)) + 1);
        return id;
    }

    /// Return the program functions in insertion order.
    ///
    /// The slice is borrowed. Do not retain it across program mutation.
    pub fn functions(self: *const Program) []const Function {
        return self.function_entries.slice().items(.function);
    }

    /// Return function identities in the same insertion order as `functions`.
    ///
    /// The slice is borrowed. Do not retain it across program mutation.
    pub fn function_ids(self: *const Program) []const FunctionId {
        return self.function_entries.slice().items(.id);
    }

    /// Replace a function without changing its identity or name.
    pub fn replace_function(
        self: *Program,
        id: FunctionId,
        func: Function,
    ) error{ FunctionIdOutOfRange, FunctionNameMismatch }!void {
        const index = self.function_index_by_id.get(id) orelse
            return error.FunctionIdOutOfRange;
        const functions_column = self.function_entries.slice().items(.function);
        if (!std.mem.eql(u8, functions_column[index].name, func.name))
            return error.FunctionNameMismatch;
        functions_column[index] = func;
    }

    /// Capture append-only state and entry selection for error cleanup.
    pub fn checkpoint_appends(self: *const Program) AppendCheckpoint {
        return .{
            .function_count = self.function_entries.len,
            .reservation_count = self.reserved_function_names.items.len,
            .entry = self.entry,
        };
    }

    /// Discard functions and name reservations added after `saved`.
    ///
    /// Function replacements and arena allocations are unaffected.
    pub fn restore_appends(self: *Program, saved: AppendCheckpoint) void {
        std.debug.assert(saved.function_count <= self.function_entries.len);
        std.debug.assert(saved.reservation_count <= self.reserved_function_names.items.len);
        const entries = self.function_entries.slice();
        const ids = entries.items(.id);
        const functions_column = entries.items(.function);
        for (saved.function_count..self.function_entries.len) |index| {
            std.debug.assert(self.function_index_by_id.remove(ids[index]));
            std.debug.assert(self.function_id_by_name.remove(functions_column[index].name));
        }
        self.function_entries.shrinkRetainingCapacity(saved.function_count);
        self.reserved_function_names.shrinkRetainingCapacity(saved.reservation_count);
        self.entry = saved.entry;
    }

    /// Look up a function identity by name.
    pub fn get_function_id(self: *const Program, name: []const u8) ?FunctionId {
        return self.function_id_by_name.get(name);
    }

    /// Look up a function by its stable program identity.
    pub fn get_function_by_id(self: *const Program, id: FunctionId) ?Function {
        const index = self.function_index_by_id.get(id) orelse return null;
        return self.function_entries.slice().items(.function)[index];
    }

    /// Select the function used by compilation entrypoints.
    pub fn set_entry(self: *Program, function: FunctionId) error{FunctionIdOutOfRange}!void {
        if (!self.function_index_by_id.contains(function)) return error.FunctionIdOutOfRange;
        self.entry = function;
    }

    /// Return the selected entry, or infer the only function not called by
    ///  another function in the program.
    pub fn resolve_entry(self: *const Program) EntryResolutionError!FunctionId {
        if (self.entry) |entry| {
            if (!self.function_index_by_id.contains(entry)) return error.EntryFunctionMissing;
            return entry;
        }
        if (self.function_entries.len == 0) return error.NoFunctions;

        var root: ?FunctionId = null;
        for (self.function_ids()) |candidate| {
            if (self.is_called_by_another_function(candidate)) continue;
            if (root != null) return error.EntrySelectionRequired;
            root = candidate;
        }
        return root orelse error.NoRootFunction;
    }

    fn is_called_by_another_function(self: *const Program, candidate: FunctionId) bool {
        for (self.functions(), self.function_ids()) |func, caller| {
            if (caller == candidate) continue;
            for (func.ops) |op| {
                if (op.params == .call and op.params.call.callee == candidate) return true;
            }
        }
        return false;
    }

    /// Allocate and reserve a function name not present in the program.
    ///
    /// The returned name is owned by `self` and is reserved against subsequent
    /// calls, including nested construction before the function is registered.
    /// The first request returns `base` when available, followed by names with
    /// increasing numeric suffixes. Reservations remain until error cleanup or
    /// program deinitialization.
    pub fn reserve_unique_function_name(self: *Program, base: []const u8) Allocator.Error![]const u8 {
        const a = self.allocator();
        var suffix: usize = 0;
        while (true) : (suffix += 1) {
            const name = if (suffix == 0)
                try a.dupe(u8, base)
            else
                try a.print("{s}_{d}", .{ base, suffix });
            if (!self.function_name_available(name)) continue;

            try self.reserved_function_names.append(a, name);
            return name;
        }
    }

    fn function_name_available(self: *const Program, name: []const u8) bool {
        if (self.function_id_by_name.contains(name)) return false;
        for (self.reserved_function_names.items) |reserved| {
            if (std.mem.eql(u8, reserved, name)) return false;
        }
        return true;
    }

    pub fn deinit(self: *Program) void {
        self.arena.deinit();
    }
};

/// Errors from function and program validation.
///
/// This includes SSA structure, operation invariants, call signatures, and
///  program-level identity checks.
/// Each op type has its own `*TypeMismatch` variant for targeted diagnostics.
pub const ValidationError = error{
    /// A function parameter names an operation as its definition.
    ParameterHasDefiningOp,
    /// One variable occupies more than one parameter or operation-output binding.
    ValueAlreadyBound,
    /// An operation output does not name that operation as its definition.
    OutputDefiningOpMismatch,
    /// An operand's owner or index does not identify its containing input slot.
    OperandOwnerMismatch,
    /// An operand refers to a value that the function does not bind.
    OperandValueNotOwned,
    /// An operand refers to an output of a later operation.
    OperandUseBeforeDefinition,
    /// A function return refers to a value that the function does not bind.
    ReturnValueNotOwned,
    UnsupportedAval,
    InvalidOpArity,
    InvalidParams,
    LiteralTypeMismatch,
    AddTypeMismatch,
    SubtractTypeMismatch,
    MultiplyTypeMismatch,
    DivideTypeMismatch,
    MaximumTypeMismatch,
    ExpTypeMismatch,
    LogTypeMismatch,
    RsqrtTypeMismatch,
    LogisticTypeMismatch,
    CompareTypeMismatch,
    SelectTypeMismatch,
    ConvertTypeMismatch,
    GatherTypeMismatch,
    ScatterTypeMismatch,
    DotTypeMismatch,
    MMTypeMismatch,
    BMMTypeMismatch,
    DotGeneralTypeMismatch,
    ConvolutionTypeMismatch,
    ReshapeTypeMismatch,
    BroadcastInDimTypeMismatch,
    TransposeTypeMismatch,
    SliceTypeMismatch,
    ConcatTypeMismatch,
    ReduceTypeMismatch,
    CallUnresolvedCallee,
    CallArityMismatch,
    CallTypeMismatch,
    CustomCallTypeMismatch,
    IotaTypeMismatch,
    DuplicateFunctionName,
    FunctionIdentityMismatch,
    EntryFunctionMissing,
    ScatterAddTypeMismatch,
    DuplicateAnnotationName,
};

/// Errors from function construction, finalization, validation, and program
///  arena allocation.
pub const BuildError = ValidationError || Allocator.Error || error{
    ParameterNotOwned,
    DuplicateParameterSelection,
    ExcludedParameterInUse,
    ExcludedParameterReturned,
};

fn same_tensor_signature(a: Tensor, b: Tensor) bool {
    if (a.dtype != b.dtype) return false;
    if (a.shape.dims.len != b.shape.dims.len) return false;
    for (a.shape.dims, 0..) |d, i| {
        if (d != b.shape.dims[i]) return false;
    }
    return true;
}

/// Validate a function's flat SSA structure and operation invariants.
///
/// - Parameters form the initial bindings. Each is unique and has no defining
///   operation.
/// - Every operand refers to a parameter or an output of a preceding operation.
/// - Each output is unique and names its containing operation as its definition.
/// - Operand owner and index fields identify the input slot containing it.
/// - Function returns refer to values bound anywhere in the function.
///
/// Operation inputs and function returns are variable references, so expressions
///  cannot nest. Together with unique, ordered bindings, these checks enforce the
///  function's A-normal form.
///
/// Operation-specific arity, parameter, and type invariants are validated after
///  the structural checks for each operation.
pub fn validate_function(func: Function) ValidationError!void {
    for (func.params, 0..) |param, index| {
        if (param.defining_op != null) {
            pr_log.debug("parameter {d} in '{s}' has a defining operation", .{ index, func.name });
            return error.ParameterHasDefiningOp;
        }
        if (value_is_bound(func.params[0..index], &.{}, param)) {
            pr_log.debug("parameter {d} in '{s}' is already bound", .{ index, func.name });
            return error.ValueAlreadyBound;
        }
    }

    for (func.ops, 0..) |op, op_index| {
        const preceding_ops = func.ops[0..op_index];
        for (op.inputs, 0..) |*operand, operand_index| {
            if (operand.owner != op or operand.index != operand_index) {
                pr_log.debug("operand {d} of operation {d} in '{s}' has inconsistent ownership", .{ operand_index, op.id, func.name });
                return error.OperandOwnerMismatch;
            }
            if (value_is_bound(func.params, preceding_ops, operand.value)) continue;
            if (value_is_bound(func.params, func.ops, operand.value)) {
                pr_log.debug("operand {d} of operation {d} in '{s}' is used before its definition", .{ operand_index, op.id, func.name });
                return error.OperandUseBeforeDefinition;
            }
            pr_log.debug("operand {d} of operation {d} in '{s}' is not owned by the function", .{ operand_index, op.id, func.name });
            return error.OperandValueNotOwned;
        }

        for (op.outputs, 0..) |output, output_index| {
            if (output.defining_op != op) {
                pr_log.debug("output {d} of operation {d} in '{s}' names a different defining operation", .{ output_index, op.id, func.name });
                return error.OutputDefiningOpMismatch;
            }
            if (value_is_bound(func.params, preceding_ops, output)) {
                pr_log.debug("output {d} of operation {d} in '{s}' is already bound", .{ output_index, op.id, func.name });
                return error.ValueAlreadyBound;
            }
            for (op.outputs[0..output_index]) |prior| {
                if (output == prior) {
                    pr_log.debug("output {d} of operation {d} in '{s}' is already bound", .{ output_index, op.id, func.name });
                    return error.ValueAlreadyBound;
                }
            }
        }
        try ops.validate(op);
    }

    for (func.returns, 0..) |return_var, index| {
        if (value_is_bound(func.params, func.ops, return_var)) continue;
        pr_log.debug("return {d} in '{s}' is not owned by the function", .{ index, func.name });
        return error.ReturnValueNotOwned;
    }
}

fn value_is_bound(params: []const *Var, function_ops: []const *Op, value: *const Var) bool {
    if (value.defining_op) |defining_op| {
        const op_index: usize = defining_op.id;
        if (op_index < function_ops.len and function_ops[op_index] == defining_op) {
            for (defining_op.outputs) |output| {
                if (output == value) return true;
            }
            return false;
        }
        for (function_ops) |op| {
            if (op != defining_op) continue;
            for (op.outputs) |output| {
                if (output == value) return true;
            }
            return false;
        }
        return false;
    }
    const param_index: usize = value.id;
    if (param_index < params.len and params[param_index] == value) return true;
    for (params) |param| {
        if (param == value) return true;
    }
    return false;
}

pub fn validate_program(program: *const Program) ValidationError!void {
    if (program.function_index_by_id.count() != program.functions().len or
        program.function_id_by_name.count() != program.functions().len)
        return error.FunctionIdentityMismatch;
    if (program.entry) |entry| {
        if (!program.function_index_by_id.contains(entry)) return error.EntryFunctionMissing;
    }
    for (program.functions(), program.function_ids(), 0..) |func, id, i| {
        if (program.get_function_id(func.name) != id or
            program.function_index_by_id.get(id) != i or
            !program.assigned_function_ids.contains(id) or
            program.next_function_id <= @backingInt(id))
            return error.FunctionIdentityMismatch;
        try validate_function(func);
        try validate_annotations(func.annotations);
        for (func.regions) |region| {
            try validate_annotations(region.annotations);
        }

        for (func.ops) |op| {
            if (op.prim() != .call) continue;

            const call_params = op.params.call;
            const callee_func = program.get_function_by_id(call_params.callee) orelse {
                pr_log.err("call references unknown function id {d} in '{s}'", .{ @backingInt(call_params.callee), func.name });
                return error.CallUnresolvedCallee;
            };

            if (op.inputs.len != callee_func.params.len or op.outputs.len != callee_func.returns.len) {
                pr_log.err("call arity mismatch: '{s}' expects {d} inputs/{d} outputs, got {d}/{d}", .{
                    callee_func.name,
                    callee_func.params.len,
                    callee_func.returns.len,
                    op.inputs.len,
                    op.outputs.len,
                });
                return error.CallArityMismatch;
            }

            for (op.inputs, 0..) |operand, idx| {
                const in_tensor = operand.value.as_tensor();
                const callee_tensor = callee_func.params[idx].as_tensor();
                if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
            for (op.outputs, 0..) |out_var, idx| {
                const out_tensor = out_var.as_tensor();
                const callee_tensor = callee_func.returns[idx].as_tensor();
                if (!same_tensor_signature(out_tensor, callee_tensor)) return error.CallTypeMismatch;
            }
        }
    }
}

fn validate_annotations(annotations: []const Annotation) ValidationError!void {
    for (annotations, 0..) |annotation, index| {
        for (annotations[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, annotation.name))
                return error.DuplicateAnnotationName;
        }
    }
}

/// Annotation stack entry for tracking active regions during building.
const RegionEntry = struct {
    id: u32,
    name: []const u8,
    annotations: []const Annotation,
    start_index: u32,
};

/// Copy annotations and their variable-length values into `allocator`.
pub fn dupe_annotations(allocator: Allocator, annotations: []const Annotation) BuildError![]const Annotation {
    const result = try allocator.alloc(Annotation, annotations.len);
    for (annotations, 0..) |annotation, index| {
        for (annotations[0..index]) |prior| {
            if (std.mem.eql(u8, prior.name, annotation.name))
                return error.DuplicateAnnotationName;
        }

        result[index] = .{
            .name = try allocator.dupe(u8, annotation.name),
            .value = switch (annotation.value) {
                .string => |value| .{ .string = try allocator.dupe(u8, value) },
                .strings => |values| strings: {
                    const copied = try allocator.alloc([]const u8, values.len);
                    for (values, copied) |value, *destination| {
                        destination.* = try allocator.dupe(u8, value);
                    }
                    break :strings .{ .strings = copied };
                },
                .bytes => |value| .{ .bytes = try allocator.dupe(u8, value) },
                else => annotation.value,
            },
        };
    }
    return result;
}

/// Mutable builder for constructing a Function.
///
/// All allocations go through the program arena.
/// Operation parameters and caller-supplied result types are copied into the
///  arena, so callers can pass stack-local slices.
///
/// Use `push_region` / `pop_region` to annotate op ranges with compilation
///  hints (outlining, kernelization). Regions are materialized at `finish`.
pub const FunctionBuilder = struct {
    /// Values and parameter policy supplied when completing a function.
    pub const FinishOptions = struct {
        /// Values returned by the completed function.
        returns: []const *Var,
        /// Parameters retained in the function signature. A selected list may
        ///  reorder parameters but must contain only distinct builder parameters.
        parameters: union(enum) {
            /// Retain every parameter in creation order.
            all,
            /// Retain the listed parameters in the supplied order.
            selected: []const *Var,
        } = .all,
    };

    const ResultTypes = union(enum) {
        /// Borrow abstract values already owned by the program.
        borrow_avals: []const Aval,
        /// Borrow abstract values from variables owned by the program.
        borrow_vars: []const *Var,
        /// Copy abstract values into the program.
        copy_avals: []const Aval,

        fn len(self: ResultTypes) usize {
            return switch (self) {
                inline else => |items| items.len,
            };
        }

        fn set(
            self: ResultTypes,
            allocator: Allocator,
            index: usize,
            destination: *Aval,
        ) Allocator.Error!void {
            destination.* = switch (self) {
                .borrow_avals => |avals| avals[index],
                .borrow_vars => |vars| vars[index].aval,
                .copy_avals => |avals| try avals[index].dupe(allocator),
            };
        }
    };

    program: *Program,
    name: []const u8,
    params_list: std.ArrayList(*Var),
    ops_list: std.ArrayList(*Op),
    next_var_id: u32,
    next_op_id: u32,
    next_region_id: u32,
    region_stack: std.ArrayList(RegionEntry),
    completed_regions: std.ArrayList(Region),

    pub fn init(program: *Program, name: []const u8) BuildError!FunctionBuilder {
        const a = program.allocator();
        return .{
            .program = program,
            .name = try a.dupe(u8, name),
            .params_list = try std.ArrayList(*Var).initCapacity(a, 8),
            .ops_list = try std.ArrayList(*Op).initCapacity(a, 16),
            .next_var_id = 0,
            .next_op_id = 0,
            .next_region_id = 0,
            .region_stack = try std.ArrayList(RegionEntry).initCapacity(a, 4),
            .completed_regions = try std.ArrayList(Region).initCapacity(a, 4),
        };
    }

    pub fn deinit(self: *FunctionBuilder) void {
        const a = self.program.allocator();
        self.params_list.deinit(a);
        self.ops_list.deinit(a);
        self.region_stack.deinit(a);
        self.completed_regions.deinit(a);
    }

    pub fn alloc(self: *FunctionBuilder) std.mem.Allocator {
        return self.program.allocator();
    }

    fn next_id(self: *FunctionBuilder) u32 {
        const id = self.next_var_id;
        self.next_var_id += 1;
        return id;
    }

    fn next_op(self: *FunctionBuilder) u32 {
        const id = self.next_op_id;
        self.next_op_id += 1;
        return id;
    }

    fn next_region(self: *FunctionBuilder) u32 {
        const id = self.next_region_id;
        self.next_region_id += 1;
        return id;
    }

    fn create_var(self: *FunctionBuilder, aval: Aval) BuildError!*Var {
        const a = self.alloc();
        const v = try a.create(Var);
        v.* = .{ .id = self.next_id(), .aval = aval };
        return v;
    }

    // Region API

    /// Push a named annotation region.
    /// Ops emitted after this call belong to this region until pop_region is called.
    pub fn push_region(self: *FunctionBuilder, name: []const u8, annotations: []const Annotation) BuildError!void {
        const a = self.alloc();
        try self.region_stack.ensureUnusedCapacity(a, 1);
        const name_copy = try a.dupe(u8, name);
        const annotations_copy = try dupe_annotations(a, annotations);
        self.region_stack.appendAssumeCapacity(.{
            .id = self.next_region(),
            .name = name_copy,
            .annotations = annotations_copy,
            .start_index = @intCast(self.ops_list.items.len),
        });
    }

    /// Pop the most recent annotation region.
    pub fn pop_region(self: *FunctionBuilder) BuildError!void {
        const a = self.alloc();
        const entry = self.region_stack.last() orelse return;
        const op_end: u32 = @intCast(self.ops_list.items.len);
        if (op_end > entry.start_index) {
            try self.completed_regions.ensureUnusedCapacity(a, 1);
            const op_ids = try a.alloc(u32, op_end - entry.start_index);
            for (self.ops_list.items[entry.start_index..op_end], 0..) |op, i| {
                op_ids[i] = op.id;
            }
            _ = self.region_stack.pop();
            self.completed_regions.appendAssumeCapacity(.{
                .id = entry.id,
                .name = entry.name,
                .annotations = entry.annotations,
                .op_ids = op_ids,
            });
        } else {
            _ = self.region_stack.pop();
        }
    }

    // Core Emit

    /// Emit a single-output op, inferring the output type via the op registry.
    ///
    /// Input vars have their use-lists updated: each input gains an Operand
    ///  node linking it to the new op.
    pub fn emit(self: *FunctionBuilder, params: Params, inputs: []const *Var) BuildError!*Var {
        const a = self.alloc();
        const stored_params = try params.dupe(a);
        const out_aval = try ops.infer_output(a, stored_params, inputs);
        const op = try self.append_op(stored_params, inputs, .{ .borrow_avals = &.{out_aval} });
        std.debug.assert(op.outputs.len == 1);
        return op.outputs[0];
    }

    /// Append an operation and return its stable program-owned address.
    ///
    /// All fallible work completes before ids or def-use chains change.
    fn append_op(
        self: *FunctionBuilder,
        /// Parameters whose nested storage belongs to the program.
        params: Params,
        /// Program-owned input variables attached to the new operation.
        inputs: []const *Var,
        /// Result abstract values and the policy for borrowing or copying them.
        result_types: ResultTypes,
    ) BuildError!*Op {
        const a = self.alloc();

        for (inputs, 0..) |input, index| {
            if (value_is_bound(self.params_list.items, self.ops_list.items, input)) continue;
            pr_log.debug("input {d} for the next operation in '{s}' is not owned by the builder", .{ index, self.name });
            return error.OperandValueNotOwned;
        }

        try self.ops_list.ensureUnusedCapacity(a, 1);
        const out_vars = try a.alloc(*Var, result_types.len());
        const vars = try a.alloc(Var, result_types.len());
        const operands = try a.alloc(Operand, inputs.len);
        const op = try a.create(Op);

        for (vars, 0..) |*out_var, index|
            try result_types.set(a, index, &out_var.aval);

        for (vars, out_vars) |*out_var, *out_var_ptr| {
            const aval = out_var.aval;
            out_var.* = .{
                .id = self.next_id(),
                .aval = aval,
                .defining_op = op,
            };
            out_var_ptr.* = out_var;
        }

        op.* = .{ .id = self.next_op(), .inputs = operands, .outputs = out_vars, .params = params };
        for (inputs, operands, 0..) |input, *operand, index|
            operand.attach(input, op, @intCast(index));

        self.ops_list.appendAssumeCapacity(op);
        return op;
    }

    /// Replay `source` as a new operation and return it.
    pub fn replay_op(
        self: *FunctionBuilder,
        /// Operation to replay. Parameter and abstract-value slices are
        ///  borrowed, so `source` must belong to the builder's program.
        source: *const Op,
        /// Builder-owned values corresponding to `source.inputs` in order.
        inputs: []const *Var,
    ) BuildError!*Op {
        if (inputs.len != source.inputs.len) return error.InvalidOpArity;
        return try self.append_op(source.params, inputs, .{ .borrow_vars = source.outputs });
    }

    /// Append a parameter with the given abstract value.
    pub fn param_like(self: *FunctionBuilder, aval: Aval) BuildError!*Var {
        const a = self.alloc();
        try self.params_list.ensureUnusedCapacity(a, 1);
        const v = try self.create_var(try aval.dupe(a));
        self.params_list.appendAssumeCapacity(v);
        return v;
    }

    pub fn param_tensor(self: *FunctionBuilder, dtype: DType, dims: []const i64) BuildError!*Var {
        return try self.param_like(.{ .tensor = .{
            .dtype = dtype,
            .shape = .{ .dims = dims },
        } });
    }

    pub fn literal_scalar(self: *FunctionBuilder, value: Literal) BuildError!*Var {
        return try self.emit(.{ .literal = value }, &.{});
    }

    /// Emit a scalar constant of `dtype` from an `f64` value.
    ///
    /// Handles dtype-aware conversion from f64. Prefer this over the two-step
    /// `Literal.from_f64` + `literal_scalar` pattern.
    pub fn scalar(self: *FunctionBuilder, dtype: DType, val: f64) BuildError!*Var {
        return try self.literal_scalar(Literal.from_f64(dtype, val));
    }

    pub fn add(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .add = {} }, &.{ lhs, rhs });
    }

    pub fn subtract(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .subtract = {} }, &.{ lhs, rhs });
    }

    pub fn multiply(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .multiply = {} }, &.{ lhs, rhs });
    }

    pub fn divide(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .divide = {} }, &.{ lhs, rhs });
    }

    pub fn maximum(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .maximum = {} }, &.{ lhs, rhs });
    }

    pub fn exp(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .exp = {} }, &.{operand_var});
    }

    pub fn log(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .log = {} }, &.{operand_var});
    }

    pub fn rsqrt(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .rsqrt = {} }, &.{operand_var});
    }

    pub fn logistic(self: *FunctionBuilder, operand_var: *Var) BuildError!*Var {
        return try self.emit(.{ .logistic = {} }, &.{operand_var});
    }

    pub fn compare(self: *FunctionBuilder, lhs: *Var, rhs: *Var, cmp: CompareParams) BuildError!*Var {
        return try self.emit(.{ .compare = cmp }, &.{ lhs, rhs });
    }

    pub fn select(self: *FunctionBuilder, cond: *Var, on_true: *Var, on_false: *Var) BuildError!*Var {
        return try self.emit(.{ .select = {} }, &.{ cond, on_true, on_false });
    }

    pub fn convert(self: *FunctionBuilder, operand_var: *Var, out_dtype: DType) BuildError!*Var {
        if (operand_var.as_tensor().dtype == out_dtype) return operand_var;
        return try self.emit(.{ .convert = out_dtype }, &.{operand_var});
    }

    pub fn gather(self: *FunctionBuilder, operand_var: *Var, indices: *Var, gp: GatherParams) BuildError!*Var {
        return try self.emit(.{ .gather = gp }, &.{ operand_var, indices });
    }

    pub fn scatter(self: *FunctionBuilder, input: *Var, indices: *Var, updates: *Var, sp: ScatterParams) BuildError!*Var {
        return try self.emit(.{ .scatter = sp }, &.{ input, indices, updates });
    }

    pub fn slice(self: *FunctionBuilder, operand_var: *Var, sp: SliceParams) BuildError!*Var {
        return try self.emit(.{ .slice = sp }, &.{operand_var});
    }

    pub fn concatenate(self: *FunctionBuilder, operands: []const *Var, axis: i64) BuildError!*Var {
        return try self.emit(.{ .concatenate = .{ .axis = axis } }, operands);
    }

    pub fn dot_general(self: *FunctionBuilder, lhs: *Var, rhs: *Var, dg: DotGeneralParams) BuildError!*Var {
        return try self.emit(.{ .dot_general = dg }, &.{ lhs, rhs });
    }

    /// Computes the scalar dot product of two equal-length vectors.
    pub fn dot(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .dot = {} }, &.{ lhs, rhs });
    }

    /// Multiplies two rank-two matrices.
    pub fn mm(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .mm = {} }, &.{ lhs, rhs });
    }

    /// Multiplies matrices over one or more identical prefix batch dimensions.
    pub fn bmm(self: *FunctionBuilder, lhs: *Var, rhs: *Var) BuildError!*Var {
        return try self.emit(.{ .bmm = {} }, &.{ lhs, rhs });
    }

    pub fn convolution(self: *FunctionBuilder, lhs: *Var, rhs: *Var, params: ConvolutionParams) BuildError!*Var {
        return try self.emit(.{ .convolution = params }, &.{ lhs, rhs });
    }

    pub fn iota(self: *FunctionBuilder, out_dtype: DType, out_dims: []const i64, iota_dim: i64) BuildError!*Var {
        return try self.emit(.{ .iota = .{
            .out_shape = out_dims,
            .out_dtype = out_dtype,
            .dimension = iota_dim,
        } }, &.{});
    }

    pub fn reshape(self: *FunctionBuilder, operand_var: *Var, out_dims: []const i64) BuildError!*Var {
        return try self.emit(.{ .reshape = .{
            .out_shape = out_dims,
        } }, &.{operand_var});
    }

    pub fn broadcast_in_dim(self: *FunctionBuilder, operand_var: *Var, out_dims: []const i64, broadcast_dimensions: []const i64) BuildError!*Var {
        return try self.emit(.{ .broadcast_in_dim = .{
            .out_shape = out_dims,
            .dimensions = broadcast_dimensions,
        } }, &.{operand_var});
    }

    /// Create a scalar constant of `dtype` with value `val`, broadcast to `dims`.
    ///
    /// Returns a scalar if `dims` is empty.
    pub fn scalar_broadcast(self: *FunctionBuilder, dtype: DType, dims: []const i64, val: f64) BuildError!*Var {
        const lit = try self.scalar(dtype, val);
        if (dims.len == 0) return lit;
        return try self.broadcast_in_dim(lit, dims, &.{});
    }

    pub fn transpose(self: *FunctionBuilder, operand_var: *Var, permutation: []const i64) BuildError!*Var {
        return try self.emit(.{ .transpose = .{
            .permutation = permutation,
        } }, &.{operand_var});
    }

    /// Reduces `operand_var` over the selected axes.
    pub fn reduce(self: *FunctionBuilder, operand_var: *Var, params: ReduceParams) BuildError!*Var {
        return try self.emit(.{ .reduce = params }, &.{operand_var});
    }

    /// Emit and return a custom call.
    ///
    /// The program owns copies of the parameters and result abstract values.
    pub fn custom_call(
        self: *FunctionBuilder,
        params: CustomCallParams,
        inputs: []const *Var,
        out_avals: []const Aval,
    ) BuildError!*Op {
        const stored_params = try (Params{ .custom_call = params }).dupe(self.alloc());
        return try self.append_op(stored_params, inputs, .{ .copy_avals = out_avals });
    }

    /// Emit and return a call to a function registered in the program.
    ///
    /// Input arity and tensor signatures must match the registered function.
    pub fn call(self: *FunctionBuilder, callee_id: FunctionId, inputs: []const *Var) BuildError!*Op {
        const callee_fn = self.program.get_function_by_id(callee_id) orelse {
            pr_log.err("call references unknown function id {d}", .{@backingInt(callee_id)});
            return error.CallUnresolvedCallee;
        };

        if (inputs.len != callee_fn.params.len) {
            pr_log.err("call arity mismatch: '{s}' expects {d} inputs, got {d}", .{ callee_fn.name, callee_fn.params.len, inputs.len });
            return error.CallArityMismatch;
        }
        for (inputs, 0..) |in_var, i| {
            const in_tensor = in_var.aval.as_tensor();
            const callee_tensor = callee_fn.params[i].aval.as_tensor();
            if (!same_tensor_signature(in_tensor, callee_tensor)) return error.CallTypeMismatch;
        }

        const stored_params = try (Params{ .call = .{ .callee = callee_id } }).dupe(self.alloc());
        return try self.append_op(stored_params, inputs, .{ .borrow_vars = callee_fn.returns });
    }

    /// Finalize and validate the function.
    ///
    /// Open regions are closed during finalization.
    ///
    /// The builder must not be used again except to call `deinit`.
    pub fn finish(
        self: *FunctionBuilder,
        /// Function signature and parameter-retention policy.
        options: FinishOptions,
    ) BuildError!Function {
        const a = self.alloc();

        for (self.ops_list.items) |op| try ops.validate(op);
        while (self.region_stack.items.len > 0) {
            try self.pop_region();
        }

        const stored_params = switch (options.parameters) {
            .all => self.params_list.items,
            .selected => |selected| try self.select_parameters(selected, options.returns),
        };
        const stored_returns = try a.dupe(*Var, options.returns);
        var func = Function{
            .name = self.name,
            .params = stored_params,
            .returns = stored_returns,
            .ops = self.ops_list.items,
            .regions = self.completed_regions.items,
            .var_count = self.next_var_id,
        };
        try validate_function(func);
        func.var_count = self.assign_var_ids(stored_params);
        self.params_list = .empty;
        self.ops_list = .empty;
        self.completed_regions = .empty;
        return func;
    }

    fn select_parameters(
        self: *FunctionBuilder,
        /// Builder parameters to retain in signature order.
        selected: []const *Var,
        /// Function returns used to reject excluded returned parameters.
        returns: []const *Var,
    ) BuildError![]*Var {
        for (selected, 0..) |candidate, index| {
            var owned = false;
            for (self.params_list.items) |param| {
                if (candidate == param) {
                    owned = true;
                    break;
                }
            }
            if (!owned) return error.ParameterNotOwned;
            for (selected[0..index]) |prior| {
                if (candidate == prior) return error.DuplicateParameterSelection;
            }
        }

        for (self.params_list.items) |param| {
            var retained = false;
            for (selected) |candidate| {
                if (param == candidate) {
                    retained = true;
                    break;
                }
            }
            if (retained) continue;
            if (param.first_use != null) return error.ExcludedParameterInUse;
            for (returns) |return_var| {
                if (param == return_var) return error.ExcludedParameterReturned;
            }
        }
        return try self.alloc().dupe(*Var, selected);
    }

    // Var ids form a dense index space used by lowering and serialization.
    fn assign_var_ids(self: *FunctionBuilder, params: []const *Var) u32 {
        var dense_id: u32 = 0;
        for (params) |param| {
            param.id = dense_id;
            dense_id += 1;
        }
        for (self.ops_list.items) |op| {
            for (op.outputs) |output| {
                output.id = dense_id;
                dense_id += 1;
            }
        }
        return dense_id;
    }
};

// Tests.

test {
    std.testing.refAllDecls(@This());
    _ = @import("pr_eval");
    _ = @import("tests/grad_check.zig");
}

test "Program.add_function rejects duplicate names" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    const function = Function{
        .name = "main",
        .params = &.{},
        .returns = &.{},
        .ops = &.{},
        .regions = &.{},
        .var_count = 0,
    };

    _ = try program.add_function(function);
    try std.testing.expectError(error.DuplicateFunctionName, program.add_function(function));
}

test "Program.add_function_with_id preserves explicit identities" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    const first = Function{
        .name = "first",
        .params = &.{},
        .returns = &.{},
        .ops = &.{},
        .regions = &.{},
        .var_count = 0,
    };
    const second = Function{
        .name = "second",
        .params = &.{},
        .returns = &.{},
        .ops = &.{},
        .regions = &.{},
        .var_count = 0,
    };

    const explicit: FunctionId = @fromBackingInt(@intCast(7));
    try std.testing.expectEqual(explicit, try program.add_function_with_id(first, explicit));
    try std.testing.expectError(
        error.DuplicateFunctionId,
        program.add_function_with_id(second, explicit),
    );
    try std.testing.expectEqual(
        @as(FunctionId, @fromBackingInt(@intCast(8))),
        try program.add_function(second),
    );
}

test "Program.reserve_unique_function_name increments occupied names" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var first = try FunctionBuilder.init(&program, "generated");
    defer first.deinit();
    _ = try program.add_function(try first.finish(.{ .returns = &.{} }));

    var second = try FunctionBuilder.init(&program, "generated_1");
    defer second.deinit();
    _ = try program.add_function(try second.finish(.{ .returns = &.{} }));

    try std.testing.expectEqualStrings("available", try program.reserve_unique_function_name("available"));
    try std.testing.expectEqualStrings("available_1", try program.reserve_unique_function_name("available"));
    try std.testing.expectEqualStrings("generated_2", try program.reserve_unique_function_name("generated"));
}

test "Program.restore discards appended functions and reservations" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var first = try FunctionBuilder.init(&program, "first");
    defer first.deinit();
    const first_id = try program.add_function(try first.finish(.{ .returns = &.{} }));
    try std.testing.expectEqual(@as(FunctionId, @fromBackingInt(@intCast(0))), first_id);

    const saved = program.checkpoint_appends();
    _ = try program.reserve_unique_function_name("reserved");
    var second = try FunctionBuilder.init(&program, "second");
    defer second.deinit();
    const second_id = try program.add_function(try second.finish(.{ .returns = &.{} }));
    try std.testing.expectEqual(@as(FunctionId, @fromBackingInt(@intCast(1))), second_id);

    program.restore_appends(saved);
    try std.testing.expectEqual(@as(usize, 1), program.functions().len);
    try std.testing.expectEqual(@as(usize, 0), program.checkpoint_appends().reservation_count);
    try std.testing.expect(program.get_function_id("first") != null);
    try std.testing.expect(program.get_function_id("second") == null);
    try std.testing.expect(program.get_function_by_id(@fromBackingInt(@intCast(1))) == null);
    try std.testing.expectError(
        error.DuplicateFunctionId,
        program.add_function_with_id(.{
            .name = "replacement",
            .params = &.{},
            .returns = &.{},
            .ops = &.{},
            .regions = &.{},
            .var_count = 0,
        }, second_id),
    );

    var third = try FunctionBuilder.init(&program, "third");
    defer third.deinit();
    const third_id = try program.add_function(try third.finish(.{ .returns = &.{} }));
    try std.testing.expectEqual(@as(FunctionId, @fromBackingInt(@intCast(2))), third_id);
    try std.testing.expect(program.get_function_by_id(second_id) == null);
}

test "Program resolves explicit and unique-root entries" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var callee_builder = try FunctionBuilder.init(&program, "callee");
    defer callee_builder.deinit();
    const callee = try program.add_function(try callee_builder.finish(.{ .returns = &.{} }));

    var caller_builder = try FunctionBuilder.init(&program, "caller");
    defer caller_builder.deinit();
    const call_op = try caller_builder.call(callee, &.{});
    try std.testing.expectEqual(@as(usize, 0), call_op.outputs.len);
    const caller = try program.add_function(try caller_builder.finish(.{ .returns = &.{} }));

    try std.testing.expectEqual(caller, try program.resolve_entry());
    try program.set_entry(callee);
    try std.testing.expectEqual(callee, try program.resolve_entry());
}

test "Program requires selection among multiple roots" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    inline for (.{ "first", "second" }) |name| {
        var builder = try FunctionBuilder.init(&program, name);
        defer builder.deinit();
        _ = try program.add_function(try builder.finish(.{ .returns = &.{} }));
    }

    try std.testing.expectError(error.EntrySelectionRequired, program.resolve_entry());
}

test "Program checkpoint restores entry selection" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var first_builder = try FunctionBuilder.init(&program, "first");
    defer first_builder.deinit();
    const first = try program.add_function(try first_builder.finish(.{ .returns = &.{} }));
    try program.set_entry(first);
    const saved = program.checkpoint_appends();

    var second_builder = try FunctionBuilder.init(&program, "second");
    defer second_builder.deinit();
    const second = try program.add_function(try second_builder.finish(.{ .returns = &.{} }));
    try program.set_entry(second);

    program.restore_appends(saved);
    try std.testing.expectEqual(first, try program.resolve_entry());
}

test "FunctionBuilder reshape validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    try std.testing.expectError(error.ReshapeTypeMismatch, b.reshape(x, &.{4}));
}

test "FunctionBuilder owns operation parameters" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const input = try builder.param_tensor(.f32, &.{6});
    var dims = [_]i64{ 2, 3 };
    const output = try builder.reshape(input, &dims);
    dims[0] = 7;

    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, output.defining_op.?.params.reshape.out_shape);
    _ = try builder.finish(.{ .returns = &.{output} });
}

test "FunctionBuilder broadcast_in_dim basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{3});
    const y = try b.broadcast_in_dim(x, &.{ 2, 3 }, &.{1});
    const func = try b.finish(.{ .returns = &.{y} });
    try validate_function(func);
}

test "FunctionBuilder transpose validation" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3, 4 });
    const y = try b.transpose(x, &.{ 2, 0, 1 });
    const func = try b.finish(.{ .returns = &.{y} });
    try validate_function(func);
}

test "FunctionBuilder reduce basic" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{ 2, 3 });
    const y = try b.reduce(x, .{ .axes = &.{0}, .operation = .sum });
    const func = try b.finish(.{ .returns = &.{y} });
    try validate_function(func);
}

test "FunctionBuilder finish preserves state after validation failure" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const rhs = try builder.param_tensor(.f32, &.{ 2, 3 });
    const output = try builder.add(lhs, rhs);
    const valid_aval = output.aval;
    output.aval = .{ .tensor = .{ .dtype = .f64, .shape = .{ .dims = &.{ 2, 3 } } } };

    try std.testing.expectError(error.AddTypeMismatch, builder.finish(.{ .returns = &.{output} }));
    try std.testing.expectEqual(@as(usize, 2), builder.params_list.items.len);
    try std.testing.expectEqual(@as(usize, 1), builder.ops_list.items.len);

    output.aval = valid_aval;
    const function = try builder.finish(.{ .returns = &.{output} });
    try std.testing.expectEqual(@as(usize, 2), function.params.len);
    try std.testing.expectEqual(@as(usize, 1), function.ops.len);
}

test "FunctionBuilder finish selects and orders parameters" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    const lhs = try builder.param_tensor(.f32, &.{2});
    _ = try builder.param_tensor(.f32, &.{2});
    const rhs = try builder.param_tensor(.f32, &.{2});
    const output = try builder.add(lhs, rhs);
    const function = try builder.finish(.{
        .returns = &.{output},
        .parameters = .{ .selected = &.{ rhs, lhs } },
    });

    try std.testing.expectEqualSlices(*Var, &.{ rhs, lhs }, function.params);
    try std.testing.expectEqual(@as(u32, 0), rhs.id);
    try std.testing.expectEqual(@as(u32, 1), lhs.id);
    try std.testing.expectEqual(@as(u32, 2), output.id);
    try std.testing.expectEqual(@as(u32, 3), function.var_count);
}

test "FunctionBuilder finish rejects invalid parameter selections" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    {
        var builder = try FunctionBuilder.init(&program, "used");
        defer builder.deinit();
        const lhs = try builder.param_tensor(.f32, &.{2});
        const rhs = try builder.param_tensor(.f32, &.{2});
        const output = try builder.add(lhs, rhs);

        try std.testing.expectError(
            error.DuplicateParameterSelection,
            builder.finish(.{
                .returns = &.{output},
                .parameters = .{ .selected = &.{ lhs, lhs } },
            }),
        );
        try std.testing.expectError(
            error.ExcludedParameterInUse,
            builder.finish(.{
                .returns = &.{output},
                .parameters = .{ .selected = &.{lhs} },
            }),
        );

        const function = try builder.finish(.{ .returns = &.{output} });
        try std.testing.expectEqual(@as(usize, 2), function.params.len);
    }

    {
        var builder = try FunctionBuilder.init(&program, "returned");
        defer builder.deinit();
        const retained = try builder.param_tensor(.f32, &.{2});
        const returned = try builder.param_tensor(.f32, &.{2});
        try std.testing.expectError(
            error.ExcludedParameterReturned,
            builder.finish(.{
                .returns = &.{returned},
                .parameters = .{ .selected = &.{retained} },
            }),
        );
    }

    {
        var owner = try FunctionBuilder.init(&program, "owner");
        defer owner.deinit();
        const owned = try owner.param_tensor(.f32, &.{2});
        var other = try FunctionBuilder.init(&program, "other");
        defer other.deinit();
        const foreign = try other.param_tensor(.f32, &.{2});
        try std.testing.expectError(
            error.ParameterNotOwned,
            owner.finish(.{
                .returns = &.{owned},
                .parameters = .{ .selected = &.{foreign} },
            }),
        );
    }
}

test "FunctionBuilder rejects values from another function" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var foreign_builder = try FunctionBuilder.init(&program, "foreign");
    defer foreign_builder.deinit();
    const foreign_param = try foreign_builder.param_tensor(.f32, &.{2});
    const foreign_result = try foreign_builder.exp(foreign_param);

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const local = try builder.param_tensor(.f32, &.{2});

    try std.testing.expectError(error.OperandValueNotOwned, builder.add(local, foreign_param));
    try std.testing.expectError(error.OperandValueNotOwned, builder.add(local, foreign_result));
    try std.testing.expectError(
        error.ReturnValueNotOwned,
        builder.finish(.{ .returns = &.{foreign_result} }),
    );

    const output = try builder.exp(local);
    _ = try builder.finish(.{ .returns = &.{output} });
}

test "FunctionBuilder finish rechecks rebound operands" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var foreign_builder = try FunctionBuilder.init(&program, "foreign");
    defer foreign_builder.deinit();
    const foreign = try foreign_builder.param_tensor(.f32, &.{2});

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{2});
    const rhs = try builder.param_tensor(.f32, &.{2});
    const output = try builder.add(lhs, rhs);
    const later = try builder.exp(lhs);
    const operand = &output.defining_op.?.inputs[0];

    operand.set(later);
    try std.testing.expectError(
        error.OperandUseBeforeDefinition,
        builder.finish(.{ .returns = &.{output} }),
    );

    operand.set(foreign);
    try std.testing.expectError(
        error.OperandValueNotOwned,
        builder.finish(.{ .returns = &.{output} }),
    );

    operand.set(lhs);
    _ = try builder.finish(.{ .returns = &.{output} });
}

test "validate_program rejects cross-function operands" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var first_builder = try FunctionBuilder.init(&program, "first");
    defer first_builder.deinit();
    const first_lhs = try first_builder.param_tensor(.f32, &.{2});
    const first_rhs = try first_builder.param_tensor(.f32, &.{2});
    const first_output = try first_builder.add(first_lhs, first_rhs);
    const first_id = try program.add_function(try first_builder.finish(.{ .returns = &.{first_output} }));

    var second_builder = try FunctionBuilder.init(&program, "second");
    defer second_builder.deinit();
    const second_param = try second_builder.param_tensor(.f32, &.{2});
    _ = try program.add_function(try second_builder.finish(.{ .returns = &.{second_param} }));

    const first = program.get_function_by_id(first_id).?;
    first.ops[0].inputs[0].set(second_param);
    try std.testing.expectError(error.OperandValueNotOwned, validate_program(&program));
}

test "validate_function rejects malformed SSA structure" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();
    const lhs = try builder.param_tensor(.f32, &.{2});
    const rhs = try builder.param_tensor(.f32, &.{2});
    const output = try builder.add(lhs, rhs);
    const function = try builder.finish(.{ .returns = &.{output} });
    const op = function.ops[0];

    lhs.defining_op = op;
    try std.testing.expectError(error.ParameterHasDefiningOp, validate_function(function));
    lhs.defining_op = null;

    function.params[1] = lhs;
    try std.testing.expectError(error.ValueAlreadyBound, validate_function(function));
    function.params[1] = rhs;

    op.inputs[0].index = 1;
    try std.testing.expectError(error.OperandOwnerMismatch, validate_function(function));
    op.inputs[0].index = 0;

    output.defining_op = null;
    try std.testing.expectError(error.OutputDefiningOpMismatch, validate_function(function));
    output.defining_op = op;

    try validate_function(function);
}

test "FunctionBuilder releases allocations after every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn build(allocator: Allocator) !void {
            var program = Program.init(allocator);
            defer program.deinit();
            var builder = try FunctionBuilder.init(&program, "main");
            defer builder.deinit();

            const lhs = try builder.param_tensor(.f32, &.{ 2, 3 });
            const rhs = try builder.param_tensor(.f32, &.{ 2, 3 });
            try builder.push_region("sum", &.{});
            const output = try builder.add(lhs, rhs);
            const function = try builder.finish(.{ .returns = &.{output} });
            _ = try program.add_function(function);
        }
    }.build, .{});

    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn build(allocator: Allocator) !void {
            var program = Program.init(allocator);
            defer program.deinit();
            var builder = try FunctionBuilder.init(&program, "selected");
            defer builder.deinit();

            const retained = try builder.param_tensor(.f32, &.{2});
            _ = try builder.param_tensor(.f32, &.{2});
            const output = try builder.exp(retained);
            const function = try builder.finish(.{
                .returns = &.{output},
                .parameters = .{ .selected = &.{retained} },
            });
            _ = try program.add_function(function);
        }
    }.build, .{});
}

test "FunctionBuilder literal scalar" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const one = try b.literal_scalar(.{ .f32 = 1.0 });
    const func = try b.finish(.{ .returns = &.{one} });
    try validate_function(func);
}

test "Literal.from_f64 tags by DType" {
    const i = Literal.from_f64(.i32, -7.0);
    const f = Literal.from_f64(.f32, 1.5);
    try std.testing.expectEqual(DType.i32, i.dtype());
    try std.testing.expectEqual(DType.f32, f.dtype());
    const g = Literal.from_f64(.f16, std.math.floatMax(f32));
    try std.testing.expectEqual(DType.f16, g.dtype());
}

test "region push/pop materializes regions" {
    const kernel = @import("kernel");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("tvm-kernel", &.{kernel.provider_annotation("tvm")});
    const d = try b.add(a_id, c_id);
    const e = try b.multiply(d, c_id);
    try b.pop_region();

    const f = try b.add(e, a_id);
    const func = try b.finish(.{ .returns = &.{f} });

    try std.testing.expectEqual(@as(usize, 1), func.regions.len);
    try std.testing.expectEqual(@as(u32, 0), func.regions[0].id);
    try std.testing.expectEqualStrings("tvm-kernel", func.regions[0].name);
    try std.testing.expectEqualStrings(
        "tvm",
        (try kernel.requested_providers(func.regions[0].annotations)).?.at(0),
    );
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, func.regions[0].op_ids);
}

test "region annotations accept namespaced values" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "annotations");
    defer builder.deinit();

    var annotation_name = [_]u8{ 'e', 'x', 'a', 'm', 'p', 'l', 'e', '.', 'i', 'd' };
    var annotation_value = [_]u8{ 'v', '1' };
    try builder.push_region("annotated", &.{.{
        .name = &annotation_name,
        .value = .{ .string = &annotation_value },
    }});
    const value = try builder.param_tensor(.f32, &.{});
    const result = try builder.log(value);
    try builder.pop_region();

    annotation_name[0] = 'X';
    annotation_value[0] = 'X';

    const func = try builder.finish(.{ .returns = &.{result} });
    const found = func.regions[0].find_annotation("example.id").?;
    try std.testing.expectEqualStrings("v1", found.value.as_string().?);
}

test "region annotation names are unique" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "duplicate_annotations");
    defer builder.deinit();

    try std.testing.expectError(error.DuplicateAnnotationName, builder.push_region("duplicate", &.{
        .{ .name = "example.value", .value = .unit },
        .{ .name = "example.value", .value = .{ .integer = 1 } },
    }));
}

test "nested regions" {
    const kernel = @import("kernel");
    const outline = @import("transform/outline.zig");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("outer", &.{kernel.provider_annotation("tvm")});
    const d = try b.add(a_id, c_id);
    {
        try b.push_region("inner", &.{outline.annotation});
        const e = try b.multiply(d, c_id);
        _ = e;
        try b.pop_region();
    }
    const f = try b.add(d, a_id);
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{f} });

    try std.testing.expectEqual(@as(usize, 2), func.regions.len);
    // Inner region completed first
    try std.testing.expectEqual(@as(u32, 1), func.regions[0].id);
    try std.testing.expectEqualStrings("inner", func.regions[0].name);
    try std.testing.expect(try outline.is_requested(func.regions[0]));
    try std.testing.expectEqualSlices(u32, &.{1}, func.regions[0].op_ids);
    // Outer region completed second
    try std.testing.expectEqual(@as(u32, 0), func.regions[1].id);
    try std.testing.expectEqualStrings("outer", func.regions[1].name);
    try std.testing.expectEqualStrings(
        "tvm",
        (try kernel.requested_providers(func.regions[1].annotations)).?.at(0),
    );
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, func.regions[1].op_ids);
}

test "regions_matching filters by predicate" {
    const kernel = @import("kernel");
    const outline = @import("transform/outline.zig");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("r1", &.{kernel.provider_annotation("tvm")});
    const d = try b.add(a_id, c_id);
    try b.pop_region();

    try b.push_region("r2", &.{outline.annotation});
    const e = try b.multiply(d, c_id);
    try b.pop_region();

    const func = try b.finish(.{ .returns = &.{e} });

    const is_kernelized = struct {
        fn f(region: Region) bool {
            return (kernel.requested_providers(region.annotations) catch null) != null;
        }
    }.f;

    var iter = func.regions_matching(is_kernelized);
    const r = iter.next().?;
    try std.testing.expectEqualStrings("r1", r.name);
    try std.testing.expect(iter.next() == null);
}

test "empty region not materialized" {
    const kernel = @import("kernel");
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const a_id = try b.param_tensor(.f32, &.{ 2, 2 });
    const c_id = try b.param_tensor(.f32, &.{ 2, 2 });

    try b.push_region("empty", &.{kernel.provider_annotation("tvm")});
    try b.pop_region();

    const d = try b.add(a_id, c_id);
    const func = try b.finish(.{ .returns = &.{d} });

    try std.testing.expectEqual(@as(usize, 0), func.regions.len);
}

test "use-list basics" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    // x is unused at this point
    try std.testing.expect(x.is_unused());

    const z = try b.add(x, y);
    // x and y each have one use now
    try std.testing.expect(x.has_one_use());
    try std.testing.expect(y.has_one_use());

    const w = try b.multiply(z, x);
    // x now has two uses (add and multiply)
    try std.testing.expect(x.has_n_uses(2));
    try std.testing.expect(!x.has_one_use());
    // z has one use
    try std.testing.expect(z.has_one_use());

    _ = try b.finish(.{ .returns = &.{w} });
}

test "FunctionBuilder assigns stable op ids" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    const z = try b.add(x, y);
    const w = try b.multiply(z, y);
    const func = try b.finish(.{ .returns = &.{w} });

    try std.testing.expectEqual(@as(u32, 0), func.ops[0].id);
    try std.testing.expectEqual(@as(u32, 1), func.ops[1].id);
    try std.testing.expectEqual(@as(u32, 0), z.defining_op.?.id);
    try std.testing.expectEqual(@as(u32, 1), w.defining_op.?.id);
}

test "Var.replace_all_uses_with" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var b = try FunctionBuilder.init(&program, "main");
    defer b.deinit();

    const x = try b.param_tensor(.f32, &.{2});
    const y = try b.param_tensor(.f32, &.{2});

    const z = try b.add(x, y);
    _ = try b.multiply(z, x);

    // x has 2 uses. Replace all uses of x with y.
    try std.testing.expect(x.has_n_uses(2));
    @constCast(x).replace_all_uses_with(@constCast(y));
    try std.testing.expect(x.is_unused());
    // y now has the uses that x had plus its original use
    try std.testing.expect(y.has_n_uses(3));

    y.replace_all_uses_with(y);
    try std.testing.expect(y.has_n_uses(3));
}

test "FunctionBuilder.custom_call owns output abstract values" {
    var program = Program.init(std.testing.allocator);
    defer program.deinit();

    var builder = try FunctionBuilder.init(&program, "main");
    defer builder.deinit();

    var dims = [_]i64{ 2, 3 };
    const call_op = try builder.custom_call(.{ .target_name = "example", .has_side_effect = false }, &.{}, &.{.{
        .tensor = .{ .dtype = .f32, .shape = .{ .dims = &dims } },
    }});
    dims[0] = 7;

    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, call_op.outputs[0].aval.as_tensor().shape.dims);
    _ = try builder.finish(.{ .returns = call_op.outputs });
}
