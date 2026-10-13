const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const with_pjrt = b.option(bool, "pjrt", "Enable PJRT backend support") orelse true;
    const with_iree = b.option(bool, "iree", "Enable IREE backend support") orelse true;
    if (!with_pjrt and !with_iree)
        std.debug.panic("llama-training requires at least one backend", .{});

    const sdk = b.option([]const u8, "sdk", "Path to Zigrad external inputs") orelse
        b.graph.environ_map.get("ZG_EXTERNAL_SDK_ROOT") orelse
        std.debug.panic("llama-training requires -Dsdk or ZG_EXTERNAL_SDK_ROOT", .{});
    const zigrad = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .sdk = resolve_absolute(b, sdk),
        .pjrt = with_pjrt,
        .mlir = true,
        .iree = with_iree,
    });

    const executable = b.addExecutable(.{
        .name = "llama-training",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{.{ .name = "zigrad", .module = zigrad.module("zigrad") }},
        }),
    });
    executable.root_module.addRPathSpecial("/run/opengl-driver/lib");
    b.installArtifact(executable);

    const run_command = b.addRunArtifact(executable);
    run_command.step.dependOn(b.getInstallStep());
    run_command.addPassthruArgs();
    b.step("run", "Run LLaMA training or inference").dependOn(&run_command.step);
}

fn resolve_absolute(b: *std.Build, path: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    const build_root = b.fmt("{f}", .{b.root});
    return std.fs.path.join(b.allocator, &.{ build_root, path }) catch
        @panic("path join failed");
}
