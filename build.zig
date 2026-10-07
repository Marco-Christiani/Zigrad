const std = @import("std");

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const test_filters = b.option([]const []const u8, "test-filter", "Select matching test names") orelse &.{};
    const strip = b.option(bool, "strip", "Omit debug information from installed executables");

    const runtime_root_opt = b.option([]const u8, "runtime", "Override runtime bundle root (dev convenience)");
    const install_runtime_link = b.option(bool, "install-runtime-link", "Create zig-out/runtime symlink (dev convenience)") orelse false;
    const version = b.option([]const u8, "version", "Version used in installed metadata") orelse "dev";

    // Integrations are enabled explicitly.
    //
    // External input contents validate selected features without selecting them.
    const use_pjrt = b.option(bool, "pjrt", "Enable PJRT backend support") orelse false;
    const use_mlir = b.option(bool, "mlir", "Enable MLIR/StableHLO lowering") orelse false;
    const use_tvm = b.option(bool, "tvm", "Enable the TVM kernel provider") orelse false;
    const use_mirage = b.option(bool, "mirage", "Enable the Mirage-backed kernel provider") orelse false;
    const use_iree = b.option(bool, "iree", "Enable the IREE integration") orelse false;
    const use_iree_embedded_elf = b.option(bool, "iree-embedded-elf", "Enable the IREE embedded ELF local-sync runtime") orelse false;
    const use_nvrtc = b.option(bool, "nvrtc", "Enable Zigrad NVRTC support") orelse false;
    const use_cuda_runtime = b.option(bool, "cuda-runtime", "Add CUDA runtime bundle paths") orelse false;
    const has_external_integration = use_pjrt or use_mlir or use_tvm or use_mirage or use_iree or use_nvrtc or use_cuda_runtime;

    // External integrations consume one input root.
    //
    // The root contains include, library, and runtime directories.
    const sdk_root = b.option([]const u8, "sdk", "Path to zigrad external SDK root (include/, lib/, runtime/)") orelse
        b.graph.environ_map.get("ZG_EXTERNAL_SDK_ROOT");
    if (has_external_integration and sdk_root == null) {
        std.debug.panic("external integrations require -Dsdk or ZG_EXTERNAL_SDK_ROOT", .{});
    }
    const sdk_include = if (sdk_root) |root| b.fmt("{s}/include", .{root}) else null;
    const sdk_lib = if (sdk_root) |root| b.fmt("{s}/lib", .{root}) else null;
    const sdk_runtime = if (sdk_root) |root| b.fmt("{s}/runtime", .{root}) else null;

    if (use_mirage and !use_nvrtc)
        std.debug.panic("-Dmirage=true requires -Dnvrtc=true", .{});

    // Emit the operation-interface coverage matrix through `@compileLog` calls.
    const emit_op_coverage = b.option(bool, "emit-op-coverage", "Emit op interface coverage at comptime (fails the build)") orelse false;

    const build_options = b.addOptions();
    build_options.addOption(bool, "has_pjrt", use_pjrt);
    build_options.addOption(bool, "has_mlir", use_mlir);
    build_options.addOption(bool, "has_tvm", use_tvm);
    build_options.addOption(bool, "has_mirage", use_mirage);
    build_options.addOption(bool, "has_iree", use_iree);
    build_options.addOption(bool, "has_iree_embedded_elf", use_iree_embedded_elf);
    build_options.addOption(bool, "has_nvrtc", use_nvrtc);
    build_options.addOption(bool, "has_cuda_runtime", use_cuda_runtime);
    build_options.addOption(bool, "emit_op_coverage", emit_op_coverage);

    const build_options_mod = build_options.createModule();
    const safetensors_zg_dep = b.dependency("safetensors_zg", .{});
    // Each module owns its files alone. Modules import one another by name,
    //  and pr, kernel, and compilation import each other.
    const pr_mod = b.addModule("pr", .{
        .root_source_file = b.path("src/pr.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    pr_mod.addImport("build_options", build_options_mod);
    const pr_eval_mod = shared_module(b, "src/pr/tests/eval.zig", target, optimize);
    const kernel_mod = shared_module(b, "src/kernel.zig", target, optimize);
    const compilation_mod = shared_module(b, "src/compilation.zig", target, optimize);
    const device_mod = shared_module(b, "src/device.zig", target, optimize);
    const dtype_mod = shared_module(b, "src/dtype.zig", target, optimize);
    const output_mod = shared_module(b, "src/output.zig", target, optimize);
    const rtti_mod = shared_module(b, "src/utils/rtti.zig", target, optimize);
    pr_mod.addImport("pr_eval", pr_eval_mod);
    pr_mod.addImport("kernel", kernel_mod);
    pr_mod.addImport("compilation", compilation_mod);
    pr_mod.addImport("device", device_mod);
    pr_mod.addImport("dtype", dtype_mod);
    pr_mod.addImport("output", output_mod);
    pr_eval_mod.addImport("pr", pr_mod);
    kernel_mod.addImport("pr", pr_mod);
    kernel_mod.addImport("device", device_mod);
    kernel_mod.addImport("dtype", dtype_mod);
    kernel_mod.addImport("rtti", rtti_mod);
    compilation_mod.addImport("device", device_mod);
    compilation_mod.addImport("rtti", rtti_mod);

    const zigrad_mod = b.addModule("zigrad", .{
        .root_source_file = b.path("src/zigrad.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    zigrad_mod.addImport("build_options", build_options_mod);
    zigrad_mod.addImport("pr", pr_mod);
    zigrad_mod.addImport("pr_eval", pr_eval_mod);
    zigrad_mod.addImport("kernel", kernel_mod);
    zigrad_mod.addImport("compilation", compilation_mod);
    zigrad_mod.addImport("device", device_mod);
    zigrad_mod.addImport("dtype", dtype_mod);
    zigrad_mod.addImport("output", output_mod);
    zigrad_mod.addImport("rtti", rtti_mod);
    zigrad_mod.addImport("safetensors_zg", safetensors_zg_dep.module("safetensors_zg"));
    zigrad_mod.addIncludePath(b.path("src"));
    if (sdk_include) |include| zigrad_mod.addIncludePath(.{ .cwd_relative = include });

    const xla_proto_modules = if (use_pjrt) modules: {
        const protobuf_dep = try b.dependencyLazy("protobuf", .{});
        const protobuf_mod = protobuf_dep.module("protobuf");
        const xla_pb_mod = b.createModule(.{
            .root_source_file = b.path("src/c/xla/proto/xla.pb.zig"),
            .imports = &.{
                .{ .name = "protobuf", .module = protobuf_mod },
            },
        });
        zigrad_mod.addImport("protobuf", protobuf_mod);
        zigrad_mod.addImport("xla_pb", xla_pb_mod);
        break :modules .{
            .protobuf = protobuf_mod,
            .xla_pb = xla_pb_mod,
        };
    } else null;

    // Translate external declarations into private modules with stable import names.
    //
    // IREE headers contain bitfields and alignment expressions that
    //  `translate-c` rejects. Its declarations are hand-written and checked
    //  against the configured headers by `src/c/iree/abi_test.zig`.
    {
        if (use_pjrt) {
            const c_pjrt = b.addTranslateC(.{
                .root_source_file = b.path("src/c/pjrt/headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_pjrt.addIncludePath(.{ .cwd_relative = sdk_include.? });
            zigrad_mod.addImport("c-pjrt", c_pjrt.createModule());
        }

        if (use_mlir) {
            const c_mlir = b.addTranslateC(.{
                .root_source_file = b.path("src/c/mlir/headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_mlir.addIncludePath(.{ .cwd_relative = sdk_include.? });
            zigrad_mod.addImport("c-mlir", c_mlir.createModule());
        }

        if (use_tvm) {
            const c_tvm = b.addTranslateC(.{
                .root_source_file = b.path("src/c/tvm/headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_tvm.addIncludePath(.{ .cwd_relative = sdk_include.? });
            zigrad_mod.addImport("c-tvm", c_tvm.createModule());
        }

        if (use_mirage) {
            const c_mirage = b.addTranslateC(.{
                .root_source_file = b.path("src/c/mirage/headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_mirage.addIncludePath(.{ .cwd_relative = sdk_include.? });
            zigrad_mod.addImport("c-mirage", c_mirage.createModule());
        }

        if (use_nvrtc) {
            const c_nvrtc = b.addTranslateC(.{
                .root_source_file = b.path("src/c/cuda/nvrtc_headers.h"),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            c_nvrtc.addIncludePath(.{ .cwd_relative = sdk_include.? });
            zigrad_mod.addImport("c-nvrtc", c_nvrtc.createModule());
        }
    }

    if (use_tvm) {
        var inactive: std.ArrayList([]const u8) = .empty;
        const translated = [_]struct { name: []const u8, enabled: bool }{
            .{ .name = "c-pjrt", .enabled = use_pjrt },     .{ .name = "c-mlir", .enabled = use_mlir },
            .{ .name = "c-mirage", .enabled = use_mirage }, .{ .name = "c-nvrtc", .enabled = use_nvrtc },
        };
        for (translated) |module| if (!module.enabled) {
            inactive.append(b.allocator, module.name) catch @panic("OOM");
        };
        const identity = @import("tools/tvm_identity.zig").derive(b, zigrad_mod, b.fmt("target={s};optimize={s};nvrtc={};pjrt={};mlir={};mirage={};iree={};cuda={}", .{
            target.result.zigTriple(b.allocator) catch @panic("OOM"), @tagName(optimize),
            use_nvrtc,                                                use_pjrt,
            use_mlir,                                                 use_mirage,
            use_iree,                                                 use_cuda_runtime,
        }), inactive.items) catch |err| std.debug.panic("TVM lowering identity failed: {s}", .{@errorName(err)});
        build_options.addOption([]const u8, "tvm_lowering_identity", identity.digest);
        const diagnostic = b.addWriteFiles();
        const manifest = diagnostic.add("tvm-source-manifest.txt", identity.manifest);
        const identity_step = b.step("tvm-identity", "Write the TVM source manifest and classified import edges");
        identity_step.dependOn(&diagnostic.step);
        const install_manifest = b.addInstallFile(manifest, "tvm-source-manifest.txt");
        identity_step.dependOn(&install_manifest.step);
    }

    const exe = b.addExecutable(.{
        .name = "zigrad",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .strip = strip,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
                .{ .name = "safetensors_zg", .module = safetensors_zg_dep.module("safetensors_zg") },
            },
        }),
    });
    exe.root_module.addIncludePath(b.path("src"));
    if (sdk_include) |include| exe.root_module.addIncludePath(.{ .cwd_relative = include });
    if (use_mlir) link_mlir_stablehlo_capi(exe, sdk_lib.?);
    if (use_iree) link_iree(zigrad_mod, sdk_lib.?, use_iree_embedded_elf);
    if (has_external_integration)
        add_runtime_bundle(b, exe, .{
            .runtime_root = runtime_root_opt orelse sdk_runtime.?,
            .install_runtime_link = install_runtime_link,
            .cuda = use_cuda_runtime,
        });

    b.installArtifact(exe);

    const cli_metadata_mod = b.createModule(.{
        .root_source_file = b.path("src/cli/render.zig"),
        .target = b.graph.host,
        .optimize = .safe,
    });
    const cli_metadata_exe = b.addExecutable(.{
        .name = "zigrad-cli-meta",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cli_meta.zig"),
            .target = b.graph.host,
            .optimize = .safe,
            .imports = &.{
                .{ .name = "zigrad_cli_metadata", .module = cli_metadata_mod },
            },
        }),
    });
    const generate_cli_metadata = b.addRunArtifact(cli_metadata_exe);
    const cli_metadata_directory = generate_cli_metadata.addOutputDirectoryArg2("cli-meta", .{});
    generate_cli_metadata.addArg(version);
    const cli_metadata_install = b.addInstallDirectory(.{
        .source_dir = cli_metadata_directory,
        .install_dir = .prefix,
        .install_subdir = "",
    });
    const cli_metadata_step = b.step("cli-meta", "Generate completions and the zigrad manpage");
    cli_metadata_step.dependOn(&cli_metadata_install.step);

    const gen_cli_meta = b.option(
        bool,
        "gen-cli-meta",
        "Install completions and the zigrad manpage",
    ) orelse false;
    if (gen_cli_meta)
        b.getInstallStep().dependOn(&cli_metadata_install.step);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    run_cmd.addPassthruArgs();
    b.step("run", "Run the Zigrad CLI").dependOn(&run_cmd.step);

    const lib_tests = b.addTest(.{
        .name = "zigrad-tests",
        .root_module = zigrad_mod,
        .filters = test_filters,
    });
    if (use_mlir) link_mlir_stablehlo_capi(lib_tests, sdk_lib.?);
    if (has_external_integration)
        add_runtime_bundle(b, lib_tests, .{
            .runtime_root = runtime_root_opt orelse sdk_runtime.?,
            .install_runtime_link = install_runtime_link,
            .cuda = use_cuda_runtime,
        });

    const run_lib_tests = b.addRunArtifact(lib_tests);
    const cli_tests = b.addTest(.{
        .name = "zigrad-cli-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/cli.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
            },
        }),
        .filters = test_filters,
    });
    if (use_mlir) link_mlir_stablehlo_capi(cli_tests, sdk_lib.?);
    if (has_external_integration)
        add_runtime_bundle(b, cli_tests, .{
            .runtime_root = runtime_root_opt orelse sdk_runtime.?,
            .install_runtime_link = install_runtime_link,
            .cuda = use_cuda_runtime,
        });
    const run_cli_tests = b.addRunArtifact(cli_tests);

    const pr_seam_tests = b.addTest(.{
        .name = "pr-seam-tests",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests/pr_seam.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
            .imports = &.{
                .{ .name = "zigrad", .module = zigrad_mod },
                .{ .name = "pr", .module = pr_mod },
            },
        }),
        .filters = test_filters,
    });
    if (use_mlir) link_mlir_stablehlo_capi(pr_seam_tests, sdk_lib.?);
    if (has_external_integration)
        add_runtime_bundle(b, pr_seam_tests, .{
            .runtime_root = runtime_root_opt orelse sdk_runtime.?,
            .install_runtime_link = install_runtime_link,
            .cuda = use_cuda_runtime,
        });
    const run_pr_seam_tests = b.addRunArtifact(pr_seam_tests);
    b.step("pr-seam", "Check that zigrad and pr share one program representation")
        .dependOn(&run_pr_seam_tests.step);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_tests.step);
    test_step.dependOn(&run_cli_tests.step);
    test_step.dependOn(&run_pr_seam_tests.step);

    const shared_test_modules = [_]struct { name: []const u8, module: *std.Build.Module }{
        .{ .name = "pr-tests", .module = pr_mod },
        .{ .name = "pr-eval-tests", .module = pr_eval_mod },
        .{ .name = "kernel-tests", .module = kernel_mod },
        .{ .name = "compilation-tests", .module = compilation_mod },
        .{ .name = "device-tests", .module = device_mod },
        .{ .name = "dtype-tests", .module = dtype_mod },
        .{ .name = "output-tests", .module = output_mod },
        .{ .name = "rtti-tests", .module = rtti_mod },
    };
    const test_compile_step = b.step("test-compile", "Build unit tests without running");
    for (shared_test_modules) |entry| {
        const tests = b.addTest(.{
            .name = entry.name,
            .root_module = entry.module,
            .filters = test_filters,
        });
        if (use_mlir) link_mlir_stablehlo_capi(tests, sdk_lib.?);
        if (has_external_integration)
            add_runtime_bundle(b, tests, .{
                .runtime_root = runtime_root_opt orelse sdk_runtime.?,
                .install_runtime_link = install_runtime_link,
                .cuda = use_cuda_runtime,
            });
        test_step.dependOn(&b.addRunArtifact(tests).step);
        test_compile_step.dependOn(&b.addInstallArtifact(tests, .{}).step);
    }

    const install_lib_tests = b.addInstallArtifact(lib_tests, .{});
    const install_cli_tests = b.addInstallArtifact(cli_tests, .{});
    test_compile_step.dependOn(&install_lib_tests.step);
    test_compile_step.dependOn(&install_cli_tests.step);

    // Emit library autodocs to zig-out/autodoc.
    const docs_obj = b.addObject(.{
        .name = "zigrad",
        .root_module = zigrad_mod,
    });
    const install_autodoc = b.addInstallDirectory(.{
        .source_dir = docs_obj.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "autodoc",
    });
    const install_autodoc_logo = b.addInstallFileWithDir(
        b.path("assets/zg-logo.svg"),
        .prefix,
        "autodoc/zg-logo.svg",
    );
    const docs_step = b.step("docs", "Emit Zig autodocs to zig-out/autodoc");
    docs_step.dependOn(&install_autodoc.step);
    docs_step.dependOn(&install_autodoc_logo.step);

    // The HLO protobuf decoder belongs to the opt-in XLA and PJRT integration.
    if (xla_proto_modules) |modules| {
        const hlo_decode_mod = b.createModule(.{
            .root_source_file = b.path("src/c/xla/hlo_decode.zig"),
            .imports = &.{
                .{ .name = "protobuf", .module = modules.protobuf },
                .{ .name = "xla_pb", .module = modules.xla_pb },
            },
        });

        const decode_hlo = b.addExecutable(.{
            .name = "decode_hlo",
            .root_module = b.createModule(.{
                .root_source_file = b.path("tools/decode_hlo.zig"),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "protobuf", .module = modules.protobuf },
                    .{ .name = "xla_pb", .module = modules.xla_pb },
                    .{ .name = "hlo_decode", .module = hlo_decode_mod },
                },
            }),
        });
        const install = b.addInstallArtifact(decode_hlo, .{});
        b.getInstallStep().dependOn(&install.step);
        b.step("decode-hlo", "Build HLO protobuf decoder tool").dependOn(&install.step);
    }

    if (use_iree) {
        const iree_runner = b.addExecutable(.{
            .name = "iree-runner",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/iree/runner.zig"),
                .target = target,
                .optimize = optimize,
                .strip = strip,
                .link_libc = true,
            }),
        });
        iree_runner.root_module.addIncludePath(.{ .cwd_relative = sdk_include.? });
        link_iree(iree_runner.root_module, sdk_lib.?, false);
        const install_iree_runner = b.addInstallArtifact(iree_runner, .{});
        b.getInstallStep().dependOn(&install_iree_runner.step);
        b.step("iree-runner", "Build minimal IREE VMFB runner").dependOn(&iree_runner.step);
        b.step("install-iree-runner", "Install minimal IREE VMFB runner").dependOn(&install_iree_runner.step);
    }
}

/// Create a module rooted at one file of the shared program-representation
///  graph. Callers add its imports after all modules exist.
fn shared_module(
    b: *std.Build,
    root: []const u8,
    target: std.Build.ResolvedTarget,
    optimize: std.lang.Optimize,
) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
}

fn link_mlir_stablehlo_capi(exe: *std.Build.Step.Compile, sdk_lib: []const u8) void {
    exe.root_module.addLibraryPath(.{ .cwd_relative = sdk_lib });

    // MLIR and StableHLO use the libstdc++ ABI.
    exe.root_module.linkSystemLibrary("stdc++", .{});

    exe.root_module.linkSystemLibrary("MLIR-C", .{});

    exe.root_module.linkSystemLibrary("StablehloCAPI", .{});
}

/// Link the IREE runtime static archives and compile the IREE C shim.
///
/// IREE compilation runs through the configured CLI. The shim exposes inline
///  functions and macros that Zig cannot import directly.
fn link_iree(mod: *std.Build.Module, sdk_lib: []const u8, embedded_elf: bool) void {
    const b = mod.owner;

    const iree_abi = b.createModule(.{
        .root_source_file = b.path("src/c/iree/runtime.zig"),
    });
    mod.addImport("iree_abi", iree_abi);

    mod.addObjectFile(.{ .cwd_relative = b.fmt("{s}/libiree_runtime_unified.a", .{sdk_lib}) });

    // Link FlatCC archives when the runtime package emits them separately.
    for ([_][]const u8{ "libflatcc_parsing.a", "libflatcc_runtime.a" }) |name| {
        const path = b.fmt("{s}/{s}", .{ sdk_lib, name });
        b.dependOnFileMetadata(.{ .cwd_relative = path });
        if (std.Io.Dir.cwd().access(b.graph.io, path, .{})) |_| {
            mod.addObjectFile(.{ .cwd_relative = path });
        } else |_| {}
    }

    const flags: []const []const u8 = if (embedded_elf)
        &.{
            "-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl",
            "-DZG_IREE_EMBEDDED_ELF=1",
        }
    else
        &.{"-DIREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl"};
    mod.addCSourceFile(.{
        .file = b.path("src/c/iree/shim.c"),
        .flags = flags,
    });
}

fn resolve_absolute_path(b: *std.Build, path: []const u8) []const u8 {
    if (std.fs.path.isAbsolute(path)) return path;
    const build_root = b.fmt("{f}", .{b.root});
    return std.fs.path.join(b.allocator, &.{ build_root, path }) catch @panic("path join failed");
}

const RuntimeBundleOptions = struct {
    runtime_root: []const u8,
    install_runtime_link: bool,
    cuda: bool,
};

fn add_runtime_bundle(b: *std.Build, exe: *std.Build.Step.Compile, options: RuntimeBundleOptions) void {
    exe.root_module.linkSystemLibrary("dl", .{});

    // Add paths for libraries installed beside the executable.
    //
    // Runtime closures supply additional search paths through the package
    //  wrapper or development shell.
    //
    // NOTE(runtime): RUNPATH does not propagate to loaded-library dependencies.
    //  Package wrappers supply their transitive paths.
    const base_rpaths = [_][]const u8{
        "$ORIGIN",
        "$ORIGIN/../lib",
    };
    inline for (base_rpaths) |path| exe.root_module.addRPathSpecial(path);

    const cuda_rpaths = [_][]const u8{
        "$ORIGIN/../runtime/nvidia/nvrtc/lib",
        "$ORIGIN/../runtime/nvidia/cublas/lib",
        "$ORIGIN/../runtime/nvidia/cudart/lib",
        "$ORIGIN/../runtime/nvidia/cudnn/lib",
        "$ORIGIN/../runtime/nvidia/cufft/lib",
        "$ORIGIN/../runtime/nvidia/cupti/lib",
        "$ORIGIN/../runtime/nvidia/cusparse/lib",
        "$ORIGIN/../runtime/nvidia/nvjitlink/lib",
        "$ORIGIN/../runtime/nvidia/nccl/lib",
        "$ORIGIN/../runtime/nvidia/nvshmem/lib",
        "$ORIGIN/../runtime/sys/lib",
        "/run/opengl-driver/lib",
    };
    if (options.cuda) {
        inline for (cuda_rpaths) |path| exe.root_module.addRPathSpecial(path);
    }

    if (!options.install_runtime_link) return;

    const runtime_root_abs = resolve_absolute_path(b, options.runtime_root);
    const link_step = b.addSystemCommand(&.{
        "bash",                                           "-eu",          "-c",
        "mkdir -p \"$1\"; ln -sfn \"$2\" \"$1/runtime\"", "runtime-link",
    });
    link_step.addDirectoryArg2(.{ .relative = .{ .base = .install_prefix } }, .{ .make_absolute = true });
    link_step.addArg(runtime_root_abs);
    link_step.has_side_effects = true;
    exe.step.dependOn(&link_step.step);
}
