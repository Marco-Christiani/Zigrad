const std = @import("std");
const zg = @import("zigrad");
const model = @import("model.zig");
const Tensor = zg.Tensor;
const parameter_field_names = @typeInfo(model.Params).@"struct".field_names;

const zero_initialized = std.StaticStringMap(void).initComptime(.{
    .{ "stem_bias", {} },
    .{ "block1_bias1", {} },
    .{ "block1_bias2", {} },
    .{ "down_bias", {} },
    .{ "block2_bias", {} },
    .{ "classifier_bias", {} },
});

/// Initialize model parameters with deterministic fan-in-scaled values.
pub fn initialize(allocator: std.mem.Allocator) !model.Params {
    var seed: u64 = 0x7a6967726164;
    return try map(model.params_spec, .{ allocator, &seed }, struct {
        fn initialize_tensor(context: struct { std.mem.Allocator, *u64 }, spec: Tensor, comptime name: []const u8) !Tensor {
            const alloc = context[0];
            var tensor = try Tensor.host(spec.dtype, spec.shape.const_slice(), .{ .alloc = alloc });
            const values = tensor.as_slice(f32);
            if (comptime zero_initialized.has(name)) {
                @memset(values, 0);
                return tensor;
            }
            var fan_in: usize = 1;
            const dims = spec.shape.const_slice();
            for (dims[0 .. dims.len - 1]) |dim| fan_in *= @intCast(dim);
            const scale = @sqrt(6.0 / @as(f32, @floatFromInt(fan_in)));
            for (values) |*value| {
                context[1].* = context[1].* *% 6364136223846793005 +% 1442695040888963407;
                const unit = @as(f32, @floatFromInt(context[1].* >> 40)) / 16_777_215.0;
                value.* = (unit * 2.0 - 1.0) * scale;
            }
            return tensor;
        }
    }.initialize_tensor);
}

/// Apply a fallible tensor operation to every parameter and clean partial results.
pub fn map(params: model.Params, context: anytype, comptime operation: anytype) !model.Params {
    var result: model.Params = undefined;
    var initialized: usize = 0;
    errdefer inline for (parameter_field_names, 0..) |field_name, index| {
        if (index < initialized) @field(result, field_name).deinit();
    };
    inline for (parameter_field_names) |field_name| {
        @field(result, field_name) = try operation(context, @field(params, field_name), field_name);
        initialized += 1;
    }
    return result;
}

/// Release every parameter tensor.
pub fn deinit(params: *model.Params) void {
    inline for (parameter_field_names) |field_name| @field(params, field_name).deinit();
}
