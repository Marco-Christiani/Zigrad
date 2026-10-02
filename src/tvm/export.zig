//! TVM runtime-module export to native shared libraries.

const std = @import("std");

const runtime = @import("../c/tvm/runtime.zig");
const TargetKind = @import("config.zig").TargetKind;
const Linker = @import("../toolchain/linker.zig").Linker;

const log = std.log.scoped(.@"zg/tvm_export");

/// Export a compiled TVM module as an ELF shared library.
pub fn export_shared(
    module: runtime.RuntimeModule,
    io: std.Io,
    allocator: std.mem.Allocator,
    output_path: [:0]const u8,
    target: TargetKind,
    linker: Linker,
) !void {
    const host_object_path = try std.fmt.allocPrintSentinel(
        allocator,
        "{s}.host.o",
        .{output_path},
        0,
    );
    defer allocator.free(host_object_path);
    defer delete_temporary_file(io, host_object_path);

    try module.write_to_file(allocator, host_object_path, "o");

    switch (target) {
        .cpu => try linker.link(io, allocator, .{
            .format = .elf_shared,
            .output = output_path,
            .objects = &.{host_object_path},
        }),
        .cuda => {
            const device_object_path = try std.fmt.allocPrintSentinel(
                allocator,
                "{s}.devc.o",
                .{output_path},
                0,
            );
            defer allocator.free(device_object_path);
            defer delete_temporary_file(io, device_object_path);

            var packed_module = try module.pack_imports_to_llvm(allocator);
            defer packed_module.deinit();
            try packed_module.write_to_file(allocator, device_object_path, "o");

            try linker.link(io, allocator, .{
                .format = .elf_shared,
                .output = output_path,
                .objects = &.{ host_object_path, device_object_path },
            });
        },
    }
    log.info("exported {s}", .{output_path});
}

fn delete_temporary_file(io: std.Io, path: []const u8) void {
    std.Io.Dir.cwd().deleteFile(io, path) catch |err| switch (err) {
        error.FileNotFound => {},
        else => log.warn("failed to remove temporary file '{s}': {s}", .{
            path,
            @errorName(err),
        }),
    };
}
