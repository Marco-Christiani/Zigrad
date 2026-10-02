//! Tensor values across tracing, host storage, and execution.
//!
//! Traced tensors refer to PR values. Device tensors refer to executor buffers.
//! Host tensors use allocated, borrowed, or memory-mapped storage. Abstract
//!  tensors carry only a data type and shape.
//!
//! Every variant exposes `dtype` and `shape` directly. Traced tensors copy these
//!  values from `Var.aval` when constructed.
const std = @import("std");
const pr = @import("pr");
const DType = @import("dtype").DType;
const Executor = @import("execution.zig");
const utils = @import("utils.zig");
const HostBuffer = utils.HostBuffer;

const Tensor = @This();

/// Element data type.
dtype: DType,

/// Logical dimensions.
shape: pr.BoundedShape,

/// Storage or tracing state.
backing: Backing,

/// Maximum rank representable by `BoundedShape`.
pub const max_rank = pr.max_rank;

/// Storage or tracing state for one tensor.
pub const Backing = union(enum) {
    /// PR value and builder used while tracing.
    traced: Traced,

    /// Executor buffer used for runtime operations.
    device: Device,

    /// CPU-resident allocated, borrowed, or mapped storage.
    host: HostBuffer,

    /// Data type and shape without storage.
    abstract: void,
};

/// PR value and builder used by a traced tensor.
pub const Traced = struct {
    /// PR value represented by this tensor.
    var_ref: *pr.Var,

    /// Builder that accepts traced operations for this value.
    builder: *pr.FunctionBuilder,
};

/// Buffer and executor used by a device tensor.
pub const Device = struct {
    /// Executor storage handle.
    buffer: Executor.Buffer,

    /// Executor that accepts `buffer`.
    executor: *Executor,
};

/// Source specification for constructing a host-backed tensor.
pub const HostSrc = union(enum) {
    /// Allocate zeroed host memory.
    alloc: std.mem.Allocator,
    /// Borrow bytes without releasing them from `deinit`.
    ///
    /// The caller keeps the bytes alive for the lifetime of this tensor.
    borrow: []const u8,
    /// Memory-map a file path read-only.
    ///
    /// `Tensor.deinit` unmaps the file.
    ///
    /// TODO(io): Accept a file handle for callers that avoid path-based I/O.
    mmap: []const u8,
};

/// Create a traced tensor from a Var.
pub fn from_var(builder: *pr.FunctionBuilder, v: *pr.Var) Tensor {
    const t = v.as_tensor();
    return .{
        .dtype = t.dtype,
        .shape = .from_slice(t.shape.dims),
        .backing = .{ .traced = .{ .var_ref = v, .builder = builder } },
    };
}

/// Create a device tensor from an executor buffer.
pub fn from_buffer(executor: *Executor, buffer_value: Executor.Buffer, dtype: DType, shape: []const i64) Tensor {
    return .{
        .dtype = dtype,
        .shape = .from_slice(shape),
        .backing = .{ .device = .{ .buffer = buffer_value, .executor = executor } },
    };
}

/// Create a traced parameter tensor.
pub fn param(builder: *pr.FunctionBuilder, dtype: DType, shape: []const i64) !Tensor {
    const v = try builder.param_tensor(dtype, shape);
    return from_var(builder, v);
}

/// Upload host data to an executor buffer.
pub fn from_host_bytes(executor: *Executor, data: []const u8, dtype: DType, shape: []const i64) Executor.Error!Tensor {
    const buffer_value = try executor.upload(data, dtype, shape);
    return from_buffer(executor, buffer_value, dtype, shape);
}

/// Create a host tensor using the selected storage source.
pub fn host(dtype: DType, shape: []const i64, src: HostSrc) !Tensor {
    const bounded = pr.BoundedShape.from_slice(shape);
    const hb: HostBuffer = switch (src) {
        .alloc => |a| try HostBuffer.init(a, bounded, dtype),
        .borrow => |bytes| blk: {
            const align_req = dtype.size_in_bytes();
            if (align_req > 1 and @intFromPtr(bytes.ptr) % align_req != 0) {
                @panic("Tensor.host: borrowed bytes are not aligned for dtype");
            }
            break :blk HostBuffer.borrow(bytes, bounded, dtype);
        },
        .mmap => |path| try HostBuffer.init_mmap(path, bounded, dtype),
    };
    return .{ .dtype = dtype, .shape = bounded, .backing = .{ .host = hb } };
}

/// Create an abstract tensor for specification purposes (shape/dtype only, no data).
pub fn abstract(dtype: DType, shape: []const i64) Tensor {
    return .{ .dtype = dtype, .shape = .from_slice(shape), .backing = .abstract };
}

fn traced_builder(self: Tensor) !*pr.FunctionBuilder {
    return switch (self.backing) {
        .traced => |t| t.builder,
        .device, .host, .abstract => error.UnsupportedAval,
    };
}

fn traced_var(self: Tensor) !*pr.Var {
    return switch (self.backing) {
        .traced => |t| t.var_ref,
        .device, .host, .abstract => error.UnsupportedAval,
    };
}

pub fn add(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.add(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn sub(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.subtract(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn mul(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.multiply(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn div(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.divide(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

/// Computes the scalar dot product of two equal-length vectors.
pub fn dot(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.dot(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

/// Multiplies two rank-two matrices.
pub fn mm(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.mm(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

/// Multiplies matrices over one or more identical prefix batch dimensions.
pub fn bmm(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.bmm(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn reshape(self: Tensor, new_dims: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.reshape(try self.traced_var(), new_dims);
    return from_var(b, v);
}

pub fn broadcast_in_dim(self: Tensor, out_dims: []const i64, broadcast_dimensions: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.broadcast_in_dim(try self.traced_var(), out_dims, broadcast_dimensions);
    return from_var(b, v);
}

pub fn transpose(self: Tensor, permutation: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.transpose(try self.traced_var(), permutation);
    return from_var(b, v);
}

pub fn exp(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.exp(try self.traced_var());
    return from_var(b, v);
}

pub fn log(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.log(try self.traced_var());
    return from_var(b, v);
}

/// Reduces this tensor over the selected axes.
pub fn reduce(self: Tensor, params: pr.ReduceParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.reduce(try self.traced_var(), params);
    return from_var(b, v);
}

pub fn max(self: Tensor, other: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.maximum(try self.traced_var(), try other.traced_var());
    return from_var(b, v);
}

pub fn rsqrt(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.rsqrt(try self.traced_var());
    return from_var(b, v);
}

pub fn logistic(self: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.logistic(try self.traced_var());
    return from_var(b, v);
}

pub fn convert(self: Tensor, out_dtype: DType) !Tensor {
    const b = try self.traced_builder();
    const v = try b.convert(try self.traced_var(), out_dtype);
    return from_var(b, v);
}

pub fn compare(self: Tensor, other: Tensor, params: pr.CompareParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.compare(try self.traced_var(), try other.traced_var(), params);
    return from_var(b, v);
}

pub fn select(self: Tensor, cond: Tensor, on_false: Tensor) !Tensor {
    const b = try self.traced_builder();
    const v = try b.select(try cond.traced_var(), try self.traced_var(), try on_false.traced_var());
    return from_var(b, v);
}

pub fn slice(self: Tensor, start_indices: []const i64, limit_indices: []const i64, strides: []const i64) !Tensor {
    const b = try self.traced_builder();
    const v = try b.slice(try self.traced_var(), .{
        .start_indices = start_indices,
        .limit_indices = limit_indices,
        .strides = strides,
    });
    return from_var(b, v);
}

pub fn concatenate(self: Tensor, others: []const Tensor, axis: i64) !Tensor {
    const b = try self.traced_builder();

    // The `*Var` slice passed to the builder is only read during op
    //  construction (`emit` copies each pointer into its own operand
    //  array), so a stack buffer is sufficient and avoids leaving a dead
    //  allocation in the program arena.
    var stack: [max_concat_operands]*pr.Var = undefined;
    const n = others.len + 1;
    if (n > stack.len) return error.TooManyConcatOperands;
    const operands = stack[0..n];

    operands[0] = try self.traced_var();
    for (others, 0..) |t, i| {
        operands[i + 1] = try t.traced_var();
    }
    const v = try b.concatenate(operands, axis);
    return from_var(b, v);
}

/// Maximum arity accepted by `concatenate` without allocation.
pub const max_concat_operands: usize = 16;

pub fn dot_general(self: Tensor, other: Tensor, params: pr.DotGeneralParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.dot_general(try self.traced_var(), try other.traced_var(), params);
    return from_var(b, v);
}

pub fn convolution(self: Tensor, kernel: Tensor, params: pr.ConvolutionParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.convolution(try self.traced_var(), try kernel.traced_var(), params);
    return from_var(b, v);
}

pub fn gather(self: Tensor, indices: Tensor, params: pr.GatherParams) !Tensor {
    const b = try self.traced_builder();
    const v = try b.gather(try self.traced_var(), try indices.traced_var(), params);
    return from_var(b, v);
}

/// Gather rows from a 2D tensor by index. `self` must be [N, hidden],
/// `indices` must be [M]. Returns [M, hidden].
pub fn gather_rows(self: Tensor, indices: Tensor) !Tensor {
    const self_shape = self.shape.const_slice();
    if (self_shape.len != 2) return error.InvalidGatherOperand;
    const indices_shape = indices.shape.const_slice();
    if (indices_shape.len != 1) return error.InvalidGatherIndices;

    const hidden = self_shape[1];
    return try self.gather(indices, .{
        .slice_sizes = &.{ 1, hidden },
        .offset_dims = &.{1},
        .collapsed_slice_dims = &.{0},
        .start_index_map = &.{0},
        .index_vector_dim = 1,
    });
}

/// Gather individual elements from a 2D tensor by [row, col] pairs.
/// `self` must be [N, M], `indices` must be [K, 2]. Returns [K].
pub fn gather_2d(self: Tensor, indices: Tensor) !Tensor {
    const self_shape = self.shape.const_slice();
    if (self_shape.len != 2) return error.InvalidGatherOperand;
    const indices_shape = indices.shape.const_slice();
    if (indices_shape.len != 2) return error.InvalidGatherIndices;
    if (indices_shape[1] != 2) return error.InvalidGatherIndices;

    return try self.gather(indices, .{
        .slice_sizes = &.{ 1, 1 },
        .offset_dims = &.{},
        .collapsed_slice_dims = &.{ 0, 1 },
        .start_index_map = &.{ 0, 1 },
        .index_vector_dim = 1,
    });
}

/// Return a mutable element view over allocated host storage.
///
/// The returned slice aliases the tensor storage. Other backing variants panic.
pub fn as_slice(self: Tensor, comptime T: type) []T {
    return switch (self.backing) {
        .host => |hb| @alignCast(std.mem.bytesAsSlice(T, switch (hb.backing) {
            .heap => |h| h.data,
            .mmap, .borrowed => @panic("as_slice requires mutable host backing (heap)"),
        })),
        .traced, .device, .abstract => @panic("as_slice requires host backing"),
    };
}

/// View host data as immutable elements of `T`. Any host backing variant.
pub fn as_const_slice(self: Tensor, comptime T: type) []const T {
    return switch (self.backing) {
        .host => |hb| @alignCast(std.mem.bytesAsSlice(T, hb.data())),
        .traced, .device, .abstract => @panic("as_const_slice requires host backing"),
    };
}

/// Raw byte slice (read-only). Host-backed only.
pub fn host_data(self: Tensor) []const u8 {
    return switch (self.backing) {
        .host => |hb| hb.data(),
        .traced, .device, .abstract => @panic("host_data requires host backing"),
    };
}

/// Raw mutable byte slice. Host-backed (heap) only.
pub fn host_data_mut(self: Tensor) []u8 {
    return switch (self.backing) {
        .host => |hb| hb.data_mut(),
        .traced, .device, .abstract => @panic("host_data_mut requires host backing"),
    };
}

/// Fill host buffer with a scalar value. Host-backed (heap) only.
pub fn fill(self: Tensor, comptime T: type, value: T) void {
    const count = self.shape.num_elements();
    const elements: []T = @alignCast(std.mem.bytesAsSlice(T, self.host_data_mut()));
    for (elements[0..count]) |*elem| elem.* = value;
}

/// Read one scalar, copying device data synchronously when needed.
///
/// Works on both host and device tensors. For device tensors, performs a
///  synchronous transfer into a stack buffer.
///
/// When `T` is a float type, decodes from the tensor's dtype via
///  `DType.decode`, e.g. `item(f32)` on a bf16 tensor decodes correctly.
/// For non-float `T`, reinterprets the raw bytes directly (caller must
///  ensure `T` matches the tensor's storage type).
///
/// The device buffer remains valid after the read.
///
pub fn item(self: Tensor, comptime T: type) !T {
    // Poison bytes not overwritten by narrow element transfers.
    var buf: [8]u8 = "\xDE\xAD\xBE\xEF\xEF\xBE\xAD\xDE".*;

    const bytes: []const u8 = switch (self.backing) {
        .host => self.host_data(),
        .device => |d| blk: {
            const nbytes = self.dtype.size_in_bytes();
            if (try d.executor.download(d.buffer, buf[0..nbytes])) |event| {
                try d.executor.wait(event);
                d.executor.release_event(event);
            }
            break :blk buf[0..nbytes];
        },
        .traced, .abstract => return error.UnsupportedAval,
    };
    return switch (self.dtype) {
        inline .f32, .f64, .bf16, .f16, .i32 => |tag| {
            return tag.decode(T, std.mem.bytesToValue(tag.StorageType(), bytes[0..@sizeOf(tag.StorageType())]));
        },
        else => @panic("item: unsupported dtype"),
    };
}

/// Copy device buffer contents to a caller-provided host byte slice.
///
/// Returns null when the transfer completed synchronously.
pub fn to_host_async(self: Tensor, dst: []u8) (Executor.Error || error{UnsupportedAval})!?Executor.Event {
    return switch (self.backing) {
        .device => |d| try d.executor.download(d.buffer, dst),
        .traced, .host, .abstract => error.UnsupportedAval,
    };
}

/// Copy device buffer to a caller-provided host byte slice, blocking until complete.
pub fn to_host_sync(self: Tensor, dst: []u8) (Executor.Error || error{UnsupportedAval})!void {
    switch (self.backing) {
        .device => |d| {
            if (try d.executor.download(d.buffer, dst)) |event| {
                try d.executor.wait(event);
                d.executor.release_event(event);
            }
        },
        .traced, .host, .abstract => return error.UnsupportedAval,
    }
}

/// Transfer a device tensor to a new host-backed tensor, blocking until complete.
///
/// The returned tensor releases its host allocation from `deinit`.
pub fn to_host(self: Tensor, allocator: std.mem.Allocator) !Tensor {
    switch (self.backing) {
        .device => |d| {
            var hb = try HostBuffer.init(allocator, self.shape, self.dtype);
            errdefer hb.deinit();
            if (try d.executor.download(d.buffer, hb.data_mut())) |event| {
                try d.executor.wait(event);
                d.executor.release_event(event);
            }
            return .{ .dtype = self.dtype, .shape = self.shape, .backing = .{ .host = hb } };
        },
        .traced, .host, .abstract => return error.UnsupportedAval,
    }
}

/// Transfer a host tensor to a device, returning a new device-backed tensor.
///
/// The source host tensor remains valid after the copy.
pub fn to_device(self: Tensor, executor: *Executor) (Executor.Error || error{UnsupportedAval})!Tensor {
    return switch (self.backing) {
        .host => |hb| {
            const buffer_value = try executor.upload(hb.data(), self.dtype, self.shape.const_slice());
            return from_buffer(executor, buffer_value, self.dtype, self.shape.const_slice());
        },
        .traced, .device, .abstract => error.UnsupportedAval,
    };
}

/// Release resources held by this tensor.
///
/// Device buffers, host allocations, and mappings are released. Borrowed, traced,
///  and abstract tensors require no release action.
pub fn deinit(self: *Tensor) void {
    switch (self.backing) {
        .device => |d| d.executor.release(d.buffer),
        .host => |*hb| hb.deinit(),
        .traced, .abstract => {},
    }
    self.* = undefined;
}

/// Return shape dims as a slice.
pub fn dims(self: *const Tensor) []const i64 {
    return self.shape.const_slice();
}

/// Shape rank.
pub fn rank(self: Tensor) usize {
    return self.shape.len;
}

/// Get the executor buffer for device storage.
pub fn buffer(self: Tensor) !Executor.Buffer {
    return switch (self.backing) {
        .device => |d| d.buffer,
        .traced, .host, .abstract => error.UnsupportedAval,
    };
}

/// Get the underlying Var (traced backing only).
pub fn get_var(self: Tensor) !*pr.Var {
    return try self.traced_var();
}

/// Create a scalar constant broadcast to match this tensor's dtype and shape.
pub fn constant_like(like: Tensor, val: f64) !Tensor {
    const b = try like.traced_builder();
    const s = from_var(b, try b.scalar(like.dtype, val));
    if (like.rank() == 0) return s;
    return try s.broadcast_in_dim(like.shape.const_slice(), &.{});
}

/// Rectified linear unit: max(x, 0). Composite: broadcast scalar zero + max.
pub fn relu(self: Tensor) !Tensor {
    return try self.max(try Tensor.constant_like(self, 0));
}
