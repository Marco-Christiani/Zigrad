const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const build_options = b.addOptions();
    build_options.step.name = "Zigrad build options";
    const build_options_module = build_options.createModule();

    // Whether to enable metadata sorting logic from the official implementation.
    build_options.addOption(
        bool,
        "enable_sort",
        b.option(bool, "enable_sort", "Whether to enable metadata sorting logic from the official implementation") orelse true,
    );

    const safetensors_zg = b.addModule("safetensors_zg", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "build_options", .module = build_options_module },
        },
    });

    const lib = b.addLibrary(.{
        .name = "safetensors_zg",
        .root_module = safetensors_zg,
        .linkage = .static,
    });

    lib.root_module.addImport("build_options", build_options_module);
    b.installArtifact(lib);

    const benchmark = b.createModule(.{
        .root_source_file = b.path("src/benchmark.zig"),
        .target = target,
        .optimize = .fast,
        .link_libc = true,
    });

    const exe = b.addExecutable(.{
        .name = "benchmark_stz",
        .root_module = benchmark,
    });
    exe.root_module.addImport("build_options", build_options_module);

    b.installArtifact(exe);
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    run_cmd.addPassthruArgs();

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    const lib_unit_tests = b.addTest(.{
        .root_module = safetensors_zg,
    });

    lib_unit_tests.root_module.addImport("build_options", build_options_module);

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_module = benchmark,
    });
    exe_unit_tests.root_module.addImport("build_options", build_options_module);

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}
