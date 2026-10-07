//! CUDA source compilation through optional NVRTC support.

const std = @import("std");
const build_options = @import("build_options");
const nvrtc_abi = if (build_options.has_nvrtc)
    @import("../c/cuda/nvrtc.zig")
else
    struct {};

pub const cuda_home_env = "CUDA_HOME";
pub const library_path_env = "ZG_NVRTC_LIBRARY_PATH";
pub const default_library_path = "libnvrtc.so";

/// Failures while resolving NVRTC host inputs.
pub const ConfigError = error{MissingCudaToolkitRoot};

/// Host and toolkit inputs shared by NVRTC consumers.
pub const Config = struct {
    /// NVRTC shared library path or soname.
    library_path: []const u8 = default_library_path,

    /// CUDA toolkit root containing `include/`.
    toolkit_root: []const u8,

    /// C library include root required by packaged CUDA headers.
    glibc_include_dir: ?[]const u8 = null,

    /// Compiler builtin include root required by packaged CUDA headers.
    gcc_include_dir: ?[]const u8 = null,

    /// Resolve NVRTC host inputs from process environment.
    pub fn from_environ(
        /// Environment map, borrowed.
        environ: *const std.process.Environ.Map,
    ) ConfigError!Config {
        return .{
            .library_path = environ.get(library_path_env) orelse default_library_path,
            .toolkit_root = environ.get(cuda_home_env) orelse
                return error.MissingCudaToolkitRoot,
            .glibc_include_dir = environ.get("NIX_GLIBC_INCLUDE"),
            .gcc_include_dir = environ.get("NIX_GCC_INCLUDE"),
        };
    }

    /// Add common NVRTC arguments followed by integration-specific options.
    fn append_options(
        self: Config,
        allocator: std.mem.Allocator,
        options: *std.ArrayList([]const u8),
        additions: CompileOptions,
    ) !void {
        try append_include(
            allocator,
            options,
            try allocator.print("{s}/include/cuda/std/detail/libcxx/include", .{self.toolkit_root}),
        );
        try append_include(
            allocator,
            options,
            try allocator.print("{s}/include", .{self.toolkit_root}),
        );
        if (self.glibc_include_dir) |path| {
            try append_include(allocator, options, path);
        }
        if (self.gcc_include_dir) |path| {
            try append_include(allocator, options, path);
        }
        for (additions.include_dirs) |path| {
            try append_include(allocator, options, path);
        }

        try options.append(
            allocator,
            try allocator.print("--gpu-architecture={s}", .{additions.gpu_arch}),
        );
        if (additions.cpp_standard) |standard| {
            try options.append(
                allocator,
                try allocator.print("--std={s}", .{standard}),
            );
        }
        if (additions.default_device) {
            try options.append(allocator, "-default-device");
        }
        for (additions.defines) |name| {
            try options.append(
                allocator,
                try allocator.print("-D{s}", .{name}),
            );
        }
        try options.appendSlice(allocator, additions.extra);
    }
};

/// Integration-specific NVRTC arguments added after shared host policy.
pub const CompileOptions = struct {
    /// GPU architecture spelling, such as `sm_86`.
    gpu_arch: []const u8,

    /// Name exposed to NVRTC diagnostics.
    program_name: []const u8 = "kernel.cu",

    /// Additional include directories.
    include_dirs: []const []const u8 = &.{},

    /// C++ language level without the `--std=` prefix.
    cpp_standard: ?[]const u8 = "c++17",

    /// Emit NVRTC device code without host entry points.
    default_device: bool = true,

    /// Preprocessor names without the `-D` prefix.
    defines: []const []const u8 = &.{},

    /// Complete compiler arguments not represented by other fields.
    extra: []const []const u8 = &.{},
};

/// Errors raised while loading the NVRTC library.
pub const LoadError = if (build_options.has_nvrtc)
    nvrtc_abi.LoadError
else
    error{NvrtcUnavailable};
/// Errors raised while constructing options or compiling CUDA source.
pub const CompileError = if (build_options.has_nvrtc)
    nvrtc_abi.CompileError || error{NvrtcArchitectureRequired}
else
    error{ NvrtcUnavailable, NvrtcArchitectureRequired, OutOfMemory };

/// Load the configured NVRTC library and resolve its required symbols.
pub fn ensure_available(config: Config) LoadError!void {
    if (comptime !build_options.has_nvrtc) return error.NvrtcUnavailable;
    try nvrtc_abi.ensure_loaded(config.library_path);
}

/// Compiles CUDA source into a PTX buffer allocated by `allocator`.
///
/// The returned buffer includes NVRTC's trailing null byte.
pub fn compile(
    allocator: std.mem.Allocator,
    source: []const u8,
    config: Config,
    options: CompileOptions,
) CompileError![]u8 {
    if (comptime !build_options.has_nvrtc) return error.NvrtcUnavailable;
    if (options.gpu_arch.len == 0) return error.NvrtcArchitectureRequired;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var compiler_options: std.ArrayList([]const u8) = .empty;
    try config.append_options(arena.allocator(), &compiler_options, options);
    return try nvrtc_abi.compile(allocator, config.library_path, source, .{
        .program_name = options.program_name,
        .compiler_options = compiler_options.items,
    });
}

fn append_include(
    allocator: std.mem.Allocator,
    options: *std.ArrayList([]const u8),
    path: []const u8,
) !void {
    try options.append(
        allocator,
        try allocator.print("--include-path={s}", .{path}),
    );
}

test Config {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var options: std.ArrayList([]const u8) = .empty;
    try (Config{
        .toolkit_root = "/cuda",
        .glibc_include_dir = "/glibc",
        .gcc_include_dir = "/gcc",
    }).append_options(allocator, &options, .{
        .gpu_arch = "sm_90",
        .include_dirs = &.{"/integration"},
        .defines = &.{"EXAMPLE"},
    });

    const expected = [_][]const u8{
        "--include-path=/cuda/include/cuda/std/detail/libcxx/include",
        "--include-path=/cuda/include",
        "--include-path=/glibc",
        "--include-path=/gcc",
        "--include-path=/integration",
        "--gpu-architecture=sm_90",
        "--std=c++17",
        "-default-device",
        "-DEXAMPLE",
    };
    try std.testing.expectEqualDeep(&expected, options.items);
}

test "Config.from_environ requires the CUDA toolkit root" {
    var environ: std.process.Environ.Map = .init(std.testing.allocator);
    defer environ.deinit();

    try std.testing.expectError(
        error.MissingCudaToolkitRoot,
        Config.from_environ(&environ),
    );

    try environ.put(cuda_home_env, "/cuda");
    try environ.put(library_path_env, "/cuda/lib/libnvrtc.so");
    const config = try Config.from_environ(&environ);
    try std.testing.expectEqualStrings("/cuda", config.toolkit_root);
    try std.testing.expectEqualStrings("/cuda/lib/libnvrtc.so", config.library_path);
}
