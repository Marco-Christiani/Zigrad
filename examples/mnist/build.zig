const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const use_pjrt = b.option(bool, "pjrt", "Enable PJRT backend support") orelse true;
    const use_mlir = b.option(bool, "mlir", "Enable MLIR/StableHLO lowering") orelse true;
    const use_tvm = b.option(bool, "tvm", "Enable the TVM kernel provider") orelse false;
    const use_mirage = b.option(bool, "mirage", "Enable the Mirage kernel provider") orelse false;
    const use_iree = b.option(bool, "iree", "Enable the IREE integration") orelse true;
    const use_nvrtc = b.option(bool, "nvrtc", "Enable NVRTC support") orelse false;
    const use_cuda_runtime = b.option(bool, "cuda-runtime", "Add CUDA runtime bundle paths") orelse false;

    const sdk_root = b.option([]const u8, "sdk", "Path to Zigrad external SDK root") orelse
        b.graph.environ_map.get("ZG_EXTERNAL_SDK_ROOT") orelse
        std.debug.panic("MNIST requires -Dsdk or ZG_EXTERNAL_SDK_ROOT", .{});
    const sdk_abs = resolve_absolute(b, sdk_root);

    const zigrad_dep = b.dependency("zigrad", .{
        .target = target,
        .optimize = optimize,
        .sdk = sdk_abs,
        .pjrt = use_pjrt,
        .mlir = use_mlir,
        .tvm = use_tvm,
        .mirage = use_mirage,
        .iree = use_iree,
        .nvrtc = use_nvrtc,
        .@"cuda-runtime" = use_cuda_runtime,
    });
    const zigrad_mod = zigrad_dep.module("zigrad");

    const exe = b.addExecutable(.{
        .name = "mnist",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
    });
    exe.root_module.addRPathSpecial("/run/opengl-driver/lib");

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run MNIST training").dependOn(&run_cmd.step);
}

fn resolve_absolute(b: *std.Build, path: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    const build_root = b.fmt("{f}", .{b.root});
    return std.fs.path.join(b.allocator, &.{ build_root, path }) catch @panic("path join failed");
}
