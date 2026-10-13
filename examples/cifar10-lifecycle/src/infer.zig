//! Run one CIFAR-10 record through the packaged IREE executable.

const std = @import("std");
const zg = @import("zigrad");
const model = @import("model.zig");
const parameters = @import("parameters.zig");
const dataset_format = @import("dataset.zig");
const build_options = @import("build_options");
const Tensor = zg.Tensor;
const parameter_field_names = @typeInfo(model.Params).@"struct".field_names;

const vmfb align(16) = @embedFile("model.vmfb").*;
const checkpoint align(8) = @embedFile("model.safetensors").*;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    if (args.len < 2 or args.len > 3) return error.ExpectedCifarRecordPath;

    const external_checkpoint = if (args.len == 3) try zg.utils.mmap_file(args[2]) else null;
    defer if (external_checkpoint) |mapped| zg.utils.munmap(mapped);
    const checkpoint_bytes = external_checkpoint orelse checkpoint[0..];
    var parameters_file = try zg.SafetensorsFile.deserialize(checkpoint_bytes, allocator);
    defer parameters_file.deinit();
    var params = try zg.from_safetensors(model.Params, &parameters_file, .{
        .allocator = allocator,
        .dtype = .f32,
    });
    defer parameters.deinit(&params);

    const record = try std.Io.Dir.cwd().readFileAlloc(
        init.io,
        args[1],
        allocator,
        .limited(dataset_format.record_size + 1),
    );
    defer allocator.free(record);
    if (record.len != dataset_format.record_size) return error.InvalidCifarRecord;
    try dataset_format.validate(record);
    var image = try Tensor.host(.f32, model.inference_images_spec.shape.const_slice(), .{ .alloc = allocator });
    defer image.deinit();
    dataset_format.decode_images(image.as_slice(f32), record, 0, 1);

    const device_construction: zg.iree.DeviceConstruction = switch (build_options.runtime) {
        .embedded_elf_sync => .embedded_elf_sync,
        .registered => .registered,
    };
    var runtime = try zg.iree.Runtime.init(
        allocator,
        device_construction,
        .{ .driver = build_options.driver },
    );
    defer runtime.deinit();
    var bytecode: zg.iree.Bytecode = .{ .borrowed = vmfb[0..] };
    var executable = try runtime.load(allocator, &bytecode, "module.main");
    defer executable.deinit();

    var params_tree = try zg.utils.Tree(Tensor).from(allocator, params);
    defer params_tree.deinit();
    var inputs: [parameter_field_names.len + 1]zg.iree.Buffer = undefined;
    var initialized: usize = 0;
    defer for (inputs[0..initialized]) |*input| input.deinit();
    for (params_tree.leaves) |param| {
        inputs[initialized] = try runtime.create_buffer(
            param.host_data(),
            .f32,
            param.shape.const_slice(),
        );
        initialized += 1;
    }
    inputs[initialized] = try runtime.create_buffer(
        image.host_data(),
        .f32,
        image.shape.const_slice(),
    );
    initialized += 1;

    var invocation = try executable.invoke(allocator, &inputs);
    defer invocation.deinit();
    if (invocation.outputs.len != 1) return error.UnexpectedOutputCount;
    var logits: [model.class_count]f32 = undefined;
    try runtime.read_buffer(&invocation.outputs[0], std.mem.sliceAsBytes(&logits));
    const predicted = argmax(&logits);
    std.debug.print("predicted={d} expected={d}\n", .{ predicted, record[0] });
}

fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |value, index| {
        if (value > values[best]) best = index;
    }
    return best;
}
