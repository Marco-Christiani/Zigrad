//! Implements the safetensors format https://huggingface.co/docs/safetensors/index
const std = @import("std");
const native_endian = @import("builtin").target.cpu.arch.endian();
const build_options = @import("build_options");

pub const Error = error{
    InvalidHeader,
    InvalidHeaderSize,
    InvalidHeaderLength,
    InvalidHeaderStart,
    InvalidHeaderEnd,
    InvalidDtype,
    InvalidTensorShape,
    InvalidOffset,
    NonSequentialOffsets,
    TensorNotFound,
    UnalignedTensor,
    AlignmentMismatch,
    IncompleteBuffer,
    BufferTooSmall,
    TensorShapeOverflow,
    HeaderOverflow,
    MetadataOverflow,
    UnsupportedDtype,
};

/// Maximum header size limit [ref](https://github.com/huggingface/safetensors/blob/7bf65ad7d56be10331dd9c15b67d82d1c5f39cc0/safetensors/src/tensor.rs#L8)
pub const MAX_HEADER_SIZE = 100_000_000;

/// Officially supported data types
pub const Dtype = enum {
    bool,
    u8,
    i8,
    f8_e5m2,
    f8_e4m3,
    i16,
    u16,
    f16,
    bf16,
    i32,
    u32,
    f32,
    f64,
    i64,
    u64,

    pub fn size(self: Dtype) usize {
        return switch (self) {
            .bool, .u8, .i8, .f8_e5m2, .f8_e4m3 => 1,
            .i16, .u16, .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }

    /// Converts a Dtype enum to its string representation.
    pub fn toString(self: Dtype) []const u8 {
        return switch (self) {
            .bool => "BOOL",
            .u8 => "U8",
            .i8 => "I8",
            .f8_e5m2 => "F8_E5M2",
            .f8_e4m3 => "F8_E4M3",
            .i16 => "I16",
            .u16 => "U16",
            .f16 => "F16",
            .bf16 => "BF16",
            .i32 => "I32",
            .u32 => "U32",
            .f32 => "F32",
            .f64 => "F64",
            .i64 => "I64",
            .u64 => "U64",
        };
    }

    pub fn fromString(str: []const u8) !Dtype {
        // produces fewer instructions than a comptime static str map, slight advantage.
        return if (std.mem.eql(u8, str, "BOOL"))
            .bool
        else if (std.mem.eql(u8, str, "U8"))
            .u8
        else if (std.mem.eql(u8, str, "I8"))
            .i8
        else if (std.mem.eql(u8, str, "F8_E5M2"))
            .f8_e5m2
        else if (std.mem.eql(u8, str, "F8_E4M3"))
            .f8_e4m3
        else if (std.mem.eql(u8, str, "I16"))
            .i16
        else if (std.mem.eql(u8, str, "U16"))
            .u16
        else if (std.mem.eql(u8, str, "F16"))
            .f16
        else if (std.mem.eql(u8, str, "BF16"))
            .bf16
        else if (std.mem.eql(u8, str, "I32"))
            .i32
        else if (std.mem.eql(u8, str, "U32"))
            .u32
        else if (std.mem.eql(u8, str, "F32"))
            .f32
        else if (std.mem.eql(u8, str, "F64"))
            .f64
        else if (std.mem.eql(u8, str, "I64"))
            .i64
        else if (std.mem.eql(u8, str, "U64"))
            .u64
        else
            Error.UnsupportedDtype;
    }
    /// For Zigrad's style convention
    pub const from_string = fromString;
    /// For Zigrad's style convention
    pub const to_string = toString;
};

/// A simple Tensor type for (de)serialization purposes.
pub const Tensor = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const usize,
    data: []align(8) const u8,
};

/// Serialize a list of Tensors into a buffer formatted as safetensors.
/// NOTE: might make more sense to accept a writer here
pub fn serializeTensors(tensors: std.ArrayList(Tensor), allocator: std.mem.Allocator) ![]u8 {
    const owns_sorted_tensors = build_options.enable_sort and tensors.items.len > 1;
    const sorted_tensors = blk: {
        if (owns_sorted_tensors) {
            const sorted_tensors = try allocator.alloc(Tensor, tensors.items.len);
            @memcpy(sorted_tensors, tensors.items);

            // Sort by dtype alignment (descending) and then by name as done in the official impl
            std.sort.insertion(Tensor, sorted_tensors, {}, struct {
                fn lessThan(_: void, a: Tensor, b: Tensor) bool {
                    const a_size = a.dtype.size();
                    const b_size = b.dtype.size();
                    if (a_size != b_size) {
                        return a_size > b_size;
                    }
                    return std.mem.lessThan(u8, a.name, b.name);
                }
            }.lessThan);
            break :blk sorted_tensors;
        } else {
            break :blk tensors.items;
        }
    };
    defer if (owns_sorted_tensors) allocator.free(sorted_tensors);

    // Get offsets for each tensor in the data section
    // offsets relative to the start of the data section
    var offset: u64 = 0;
    var aw: std.Io.Writer.Allocating = .init(allocator);
    defer aw.deinit();
    const writer = &aw.writer;
    try writer.writeAll("{");
    var first = true;
    for (sorted_tensors) |tensor| {
        if (!first) try writer.writeAll(",");
        first = false;

        const end_offset = offset + tensor.data.len;

        // tensor name
        try writer.writeAll("\"");
        if (tensor.name.len == 0) return Error.InvalidTensorShape;
        try writer.writeAll(tensor.name);
        try writer.writeAll("\":{");

        // dtype
        try writer.print("\"dtype\":\"{s}\",", .{tensor.dtype.toString()});

        // shape
        try writer.writeAll("\"shape\":[");
        for (tensor.shape, 0..) |dim, i| {
            if (i != 0) try writer.writeAll(", ");
            try writer.print("{}", .{dim});
        }
        try writer.writeAll("],");

        // data_offsets
        // brace is doubled so it's treated as a literal
        try writer.print("\"data_offsets\":[{},{}]}}", .{ offset, end_offset });
        offset = end_offset;
    }
    try writer.writeAll("}");

    const header = aw.writer.buffered();
    // extend length the next align(8) boundary, which will be where data starts
    const json_len = header.len;
    const padded_len = (json_len + 7) & ~(@as(usize, 7)); // nearest multiple of 8

    // total file size = 8 (header size) + padded header + all tensor data.
    const totalSize = 8 + padded_len + offset;
    var out_buffer = try allocator.alloc(u8, totalSize);

    std.mem.writePackedInt(u64, out_buffer[0..8], 0, json_len, native_endian);

    // Copy header and pad with spaces
    @memcpy(out_buffer[8 .. 8 + json_len], header);
    if (padded_len > json_len) {
        @memset(out_buffer[8 + json_len .. 8 + padded_len], ' ');
    }

    // Append tensor data.
    var current_offset: usize = 8 + padded_len;
    for (sorted_tensors) |tensor| {
        @memcpy(out_buffer[current_offset .. current_offset + tensor.data.len], tensor.data);
        current_offset += tensor.data.len;
    }

    return out_buffer;
}

/// For Zigrad's style convention
pub const serialize_tensors = serializeTensors;

pub const TensorInfo = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const usize,
    data_offset: struct {
        start: usize,
        end: usize,
    },

    pub fn deinit(self: *const TensorInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.shape);
    }
};

/// View into a tensor (after loading)
pub const TensorView = struct {
    info: TensorInfo,
    data: []align(8) const u8,
};

/// SafeTensors container representing a loaded file.
pub const SafeTensorsFile = struct {
    const Self = @This();
    allocator: std.mem.Allocator,
    header_size: usize,
    tensors: []TensorInfo,
    raw_data: []align(8) const u8,

    pub fn deinit(self: *Self) void {
        for (self.tensors) |*tensor| {
            tensor.deinit(self.allocator);
        }
        self.allocator.free(self.tensors);
    }

    /// Get a tensor by name, errors if not found.
    pub fn get(self: Self, name: []const u8) !TensorView {
        for (self.tensors) |info| {
            if (std.mem.eql(u8, info.name, name)) {
                const start = info.data_offset.start + 8 + self.header_size;
                const end = info.data_offset.end + 8 + self.header_size;
                if (end > self.raw_data.len) return Error.IncompleteBuffer;

                // Verify alignment
                const data_start_addr = @intFromPtr(self.raw_data.ptr) + start;
                if (data_start_addr % 8 != 0) return Error.UnalignedTensor;

                return TensorView{
                    .info = info,
                    .data = @alignCast(self.raw_data[start..end]),
                };
            }
        }
        return Error.TensorNotFound;
    }

    /// Deserialize a SafeTensors file from a byte buffer.
    pub fn deserialize(data: []const u8, allocator: std.mem.Allocator) !Self {
        if (data.len < 8) return Error.BufferTooSmall;
        const header_size = std.mem.readPackedInt(u64, data[0..8], 0, native_endian);
        if (header_size > MAX_HEADER_SIZE) return Error.HeaderOverflow;
        const header_end = 8 + header_size;
        if (header_end > data.len) return Error.BufferTooSmall;

        // Verify the data buffer is 8-byte aligned
        const data_addr = @intFromPtr(data.ptr);
        if (data_addr % 8 != 0) return Error.UnalignedTensor;

        // Calculate padded header size (data starts at 8-byte aligned boundary)
        const padded_header_size = (header_size + 7) & ~(@as(usize, 7));

        const tensors = try parseHeader(data[8..header_end], allocator);
        return Self{
            .allocator = allocator,
            .header_size = padded_header_size,
            .tensors = tensors,
            .raw_data = @alignCast(data),
        };
    }
};

/// Parses JSON header from a slice into tensor metadata
fn parseHeader(json: []const u8, allocator: std.mem.Allocator) ![]TensorInfo {
    var tensors = std.ArrayList(TensorInfo).empty;
    defer tensors.deinit(allocator);

    var scanner = JsonScanner.init(json);
    try scanner.expectToken(.ObjectBegin);
    while (true) {
        if (scanner.peek() == .ObjectEnd) {
            _ = try scanner.nextToken(); // consume '}'
            break;
        }

        const token = try scanner.nextToken();
        if (token != .String) return Error.InvalidHeader;
        const tensor_name = scanner.tokenString();

        if (std.mem.eql(u8, tensor_name, "__metadata__")) {
            try scanner.expectToken(.Colon);
            try scanner.skipValue();

            if (scanner.peek() == .Comma) {
                _ = try scanner.nextToken(); // consume ','
            } else if (scanner.peek() == .ObjectEnd) {
                _ = try scanner.nextToken(); // consume '}'
                break;
            } else {
                return Error.InvalidHeader;
            }

            continue;
        }

        // Process tensor entry
        try scanner.expectToken(.Colon);
        try scanner.expectToken(.ObjectBegin);

        var dtype: ?Dtype = null;
        var shape: ?[]usize = null;
        var start_offset: ?usize = null;
        var end_offset: ?usize = null;

        // Parse tensor properties
        while (true) {
            if (scanner.peek() == .ObjectEnd) {
                _ = try scanner.nextToken(); // consume '}'
                break;
            }

            const prop_token = try scanner.nextToken();
            if (prop_token != .String) return Error.InvalidHeader;

            const prop_name = scanner.tokenString();
            try scanner.expectToken(.Colon);

            if (std.mem.eql(u8, prop_name, "dtype")) {
                const dtype_token = try scanner.nextToken();
                if (dtype_token != .String) return Error.InvalidHeader;
                dtype = try Dtype.fromString(scanner.tokenString());
            } else if (std.mem.eql(u8, prop_name, "shape")) {
                try scanner.expectToken(.ArrayBegin);

                // Count and parse dims
                var dimensions = std.ArrayList(usize).empty;
                defer dimensions.deinit(allocator);
                if (scanner.peek() != .ArrayEnd) { // array isnt empty
                    while (true) {
                        const dim_token = try scanner.nextToken();
                        if (dim_token != .Number) return Error.InvalidHeader;

                        // NOTE: while the parser supports scientific notation, this wont.
                        try dimensions.append(allocator, try std.fmt.parseInt(usize, scanner.tokenString(), 10));

                        if (scanner.peek() == .ArrayEnd) {
                            _ = try scanner.nextToken(); // consume ']'
                            break;
                        }

                        try scanner.expectToken(.Comma);
                    }
                } else {
                    _ = try scanner.nextToken(); // consume ']'
                }
                shape = try dimensions.toOwnedSlice(allocator);
            } else if (std.mem.eql(u8, prop_name, "data_offsets")) {
                try scanner.expectToken(.ArrayBegin);

                // Get start offset
                const start_token = try scanner.nextToken();
                if (start_token != .Number) return Error.InvalidHeader;
                start_offset = try std.fmt.parseInt(usize, scanner.tokenString(), 10);

                try scanner.expectToken(.Comma);

                // Get end offset
                const end_token = try scanner.nextToken();
                if (end_token != .Number) return Error.InvalidHeader;
                end_offset = try std.fmt.parseInt(usize, scanner.tokenString(), 10);

                try scanner.expectToken(.ArrayEnd);
            } else { // unknown prop
                std.debug.print("Unexpected property: {s}@0x{x}\n", .{ scanner.tokenString(), scanner.pos });
                try scanner.skipValue();
            }

            // valid to find either a comma or end of object, I think?
            if (scanner.peek() == .Comma) {
                _ = try scanner.nextToken(); // consume ','
            } else if (scanner.peek() != .ObjectEnd) {
                return Error.InvalidHeader;
            }
        }

        // Validate tensor info
        if (dtype == null or shape == null or start_offset == null or end_offset == null) {
            std.debug.print("{?} {?any} {?any} {?any}\n", .{ dtype, shape, start_offset, end_offset });
            return Error.InvalidHeader;
        }

        // Create tensor info
        try tensors.append(allocator, TensorInfo{
            .name = try allocator.dupe(u8, tensor_name),
            .dtype = dtype.?,
            .shape = shape.?, // Already owned
            .data_offset = .{
                .start = start_offset.?,
                .end = end_offset.?,
            },
        });

        if (scanner.peek() == .Comma) {
            _ = try scanner.nextToken(); // consume ','
        } else if (scanner.peek() == .ObjectEnd) {
            _ = try scanner.nextToken(); // consume '}'
            break;
        } else {
            return Error.InvalidHeader;
        }
    }
    if (build_options.enable_sort) {
        std.sort.insertion(TensorInfo, tensors.items, {}, struct {
            fn lessThan(_: void, a: TensorInfo, b: TensorInfo) bool {
                return a.data_offset.start < b.data_offset.start;
            }
        }.lessThan);
    }
    return tensors.toOwnedSlice(allocator);
}

/// For Zigrad's style convention
pub const parse_header = parseHeader;

/// Token types for `JsonScanner`
const TokenType = enum {
    ObjectBegin, // '{'
    ObjectEnd, // '}'
    ArrayBegin, // '['
    ArrayEnd, // ']'
    Colon, // ':'
    Comma, // ','
    String, // '"..."'
    Number,
    True, // 'true'
    False, // 'false'
    Null, // 'null'
    End, // End of input
};

/// Minimal json scanner for parsing header
const JsonScanner = struct {
    input: []const u8,
    pos: usize = 0,
    token_start: usize = 0,
    token_end: usize = 0,
    current_token: TokenType = .End,

    fn init(input: []const u8) JsonScanner {
        return .{
            .input = input,
        };
    }

    fn peek(self: *@This()) TokenType {
        self.skipWhitespace();
        if (self.pos >= self.input.len) return .End;

        return switch (self.input[self.pos]) {
            '{' => .ObjectBegin,
            '}' => .ObjectEnd,
            '[' => .ArrayBegin,
            ']' => .ArrayEnd,
            ':' => .Colon,
            ',' => .Comma,
            '"' => .String,
            't' => if (self.pos + 3 < self.input.len and
                std.mem.eql(u8, self.input[self.pos .. self.pos + 4], "true")) .True else @panic("InvalidHeader"), // return Error.InvalidHeader,
            'f' => if (self.pos + 4 < self.input.len and
                std.mem.eql(u8, self.input[self.pos .. self.pos + 5], "false")) .False else @panic("InvalidHeader"),
            'n' => if (self.pos + 3 < self.input.len and
                std.mem.eql(u8, self.input[self.pos .. self.pos + 4], "null")) .Null else @panic("InvalidHeader"),
            '0'...'9', '-' => .Number,
            else => @panic("InvalidHeader"), // return Error.InvalidHeader,
        };
    }

    fn nextToken(self: *@This()) !TokenType {
        self.skipWhitespace();
        if (self.pos >= self.input.len) {
            self.current_token = .End;
            return .End;
        }

        self.token_start = self.pos;

        const c = self.input[self.pos];
        self.pos += 1;

        self.current_token = blk: {
            switch (c) {
                '{' => {
                    self.token_end = self.pos;
                    break :blk .ObjectBegin;
                },
                '}' => {
                    self.token_end = self.pos;
                    break :blk .ObjectEnd;
                },
                '[' => {
                    self.token_end = self.pos;
                    break :blk .ArrayBegin;
                },
                ']' => {
                    self.token_end = self.pos;
                    break :blk .ArrayEnd;
                },
                ':' => {
                    self.token_end = self.pos;
                    break :blk .Colon;
                },
                ',' => {
                    self.token_end = self.pos;
                    break :blk .Comma;
                },
                '"' => {
                    // Parse string
                    while (self.pos < self.input.len) {
                        if (self.input[self.pos] == '"' and self.input[self.pos - 1] != '\\') {
                            self.pos += 1;
                            self.token_end = self.pos;
                            break :blk .String;
                        }
                        self.pos += 1;
                    }
                    return Error.InvalidHeader;
                },
                't' => {
                    if (self.pos + 3 <= self.input.len and
                        std.mem.eql(u8, self.input[self.pos - 1 .. self.pos + 3], "true"))
                    {
                        self.pos += 3;
                        self.token_end = self.pos;
                        break :blk .True;
                    }
                    return Error.InvalidHeader;
                },
                'f' => {
                    if (self.pos + 4 <= self.input.len and
                        std.mem.eql(u8, self.input[self.pos - 1 .. self.pos + 4], "false"))
                    {
                        self.pos += 4;
                        self.token_end = self.pos;
                        break :blk .False;
                    }
                    return Error.InvalidHeader;
                },
                'n' => {
                    if (self.pos + 3 <= self.input.len and
                        std.mem.eql(u8, self.input[self.pos - 1 .. self.pos + 3], "null"))
                    {
                        self.pos += 3;
                        self.token_end = self.pos;
                        break :blk .Null;
                    }
                    return Error.InvalidHeader;
                },
                '0'...'9', '-' => {
                    // Parse number
                    while (self.pos < self.input.len and
                        ((self.input[self.pos] >= '0' and self.input[self.pos] <= '9') or
                            self.input[self.pos] == '.' or self.input[self.pos] == 'e' or
                            self.input[self.pos] == 'E' or self.input[self.pos] == '+' or
                            self.input[self.pos] == '-'))
                    {
                        self.pos += 1;
                    }
                    self.token_end = self.pos;
                    break :blk .Number;
                },
                else => return Error.InvalidHeader,
            }
        };
        return self.current_token;
    }

    fn skipWhitespace(self: *JsonScanner) void {
        while (self.pos < self.input.len and
            (self.input[self.pos] == ' ' or self.input[self.pos] == '\t' or
                self.input[self.pos] == '\n' or self.input[self.pos] == '\r'))
        {
            self.pos += 1;
        }
    }

    fn tokenString(self: *JsonScanner) []const u8 {
        return switch (self.current_token) {
            .String => self.input[self.token_start + 1 .. self.token_end - 1], // Remove quotes
            else => self.input[self.token_start..self.token_end],
        };
    }

    fn expectToken(self: *JsonScanner, expected: TokenType) !void {
        const token = try self.nextToken();
        if (token != expected) return Error.InvalidHeader;
    }

    fn skipValue(self: *JsonScanner) !void {
        const token = try self.nextToken();
        switch (token) {
            .ObjectBegin => {
                // Skip object contents
                var depth: usize = 1;
                while (depth > 0 and self.pos < self.input.len) {
                    const next = try self.nextToken();
                    switch (next) {
                        .ObjectBegin => depth += 1,
                        .ObjectEnd => depth -= 1,
                        .String => {
                            if (depth == 1) {
                                try self.expectToken(.Colon);
                                try self.skipValue();

                                if (self.peek() == .Comma) {
                                    _ = try self.nextToken();
                                }
                            }
                        },
                        else => {},
                    }
                }
            },
            .ArrayBegin => {
                // Skip array contents
                var depth: usize = 1;
                while (depth > 0 and self.pos < self.input.len) {
                    const next = try self.nextToken();
                    switch (next) {
                        .ArrayBegin => depth += 1,
                        .ArrayEnd => depth -= 1,
                        else => {
                            if (depth == 1 and self.peek() == .Comma) {
                                _ = try self.nextToken();
                            }
                        },
                    }
                }
            },
            else => {}, // its a simple val and we already consumed it
        }
    }
};

test "serialize one caller-owned tensor" {
    const testing = std.testing;
    const shape = [_]usize{3};
    const data align(8) = [_]u8{ 1, 2, 3 };
    var backing = [_]Tensor{.{
        .name = "value",
        .dtype = .u8,
        .shape = &shape,
        .data = &data,
    }};
    const tensors = std.ArrayList(Tensor).fromOwnedSlice(&backing);

    const bytes = try serializeTensors(tensors, testing.allocator);
    defer testing.allocator.free(bytes);
    var file = try SafeTensorsFile.deserialize(bytes, testing.allocator);
    defer file.deinit();
    const view = try file.get("value");
    try testing.expectEqualSlices(u8, &data, view.data);
}

test "deserialize" {
    const allocator = std.testing.allocator;

    const header =
        \\{
        \\  "weights": {
        \\    "dtype": "F32",
        \\    "shape": [2, 2],
        \\    "data_offsets": [0, 16]
        \\  }
        \\}
    ;

    // Calculate total size with proper alignment
    const header_size = header.len;
    const padded_header_size = (header_size + 7) & ~(@as(usize, 7)); // align to 8 bytes
    const data_size = 4 * @sizeOf(f32); // 2x2 f32 tensor
    const total_size = 8 + padded_header_size + data_size;

    // Allocate aligned buffer
    var aligned_buffer = try allocator.alignedAlloc(u8, .@"8", total_size);
    defer allocator.free(aligned_buffer);

    // Write header size
    std.mem.writePackedInt(u64, aligned_buffer[0..8], 0, header_size, native_endian);

    // Write header and pad to 8-byte boundary
    @memcpy(aligned_buffer[8 .. 8 + header_size], header);
    if (padded_header_size > header_size) {
        @memset(aligned_buffer[8 + header_size .. 8 + padded_header_size], ' ');
    }

    // Write tensor data (zeros)
    @memset(aligned_buffer[8 + padded_header_size ..], 0);

    var tensors = try SafeTensorsFile.deserialize(aligned_buffer, allocator);
    defer tensors.deinit();

    const tensor = try tensors.get("weights");
    try std.testing.expectEqual(tensor.info.dtype, .f32);
    try std.testing.expectEqualSlices(usize, &[_]usize{ 2, 2 }, tensor.info.shape);
}
