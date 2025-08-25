//! llm.zig - Entry point for the LLM app
//!
//! Copyright 2025 Joe

const std = @import("std");
const build_options = @import("build_options");
const downloadModel = @import("utils.zig").downloadModel;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const LlamaTransformer = @import("llama.zig").Transformer;
const PhiTransformer = @import("phi.zig").Transformer;
const QwenTransformer = @import("qwen.zig").Transformer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);
    var chat_inputs: std.ArrayList([]const u8) = .empty;
    defer chat_inputs.deinit(allocator);
    var config_name: ?[]const u8 = null;
    var model_type_: ?[]const u8 = null;
    var model_name_: ?[]const u8 = null;
    var num_tokens_: ?usize = null;
    var chat_format_: ?[]const u8 = null;
    for (process_args) |arg| {
        if (std.mem.startsWith(u8, arg, "--")) {
            if (std.mem.indexOf(u8, arg, "=")) |equals_pos| {
                const key = arg[2..equals_pos];
                const value = arg[equals_pos + 1 ..];
                if (std.mem.eql(u8, key, "config")) {
                    config_name = value;
                } else if (std.mem.eql(u8, key, "model-type")) {
                    model_type_ = value;
                } else if (std.mem.eql(u8, key, "model-name")) {
                    model_name_ = value;
                } else if (std.mem.eql(u8, key, "format")) {
                    chat_format_ = value;
                } else if (std.mem.eql(u8, key, "max")) {
                    num_tokens_ = try std.fmt.parseInt(usize, value, 10);
                }
            } else if (std.mem.eql(u8, arg[2..], "help")) {
                printUsage();
                std.process.exit(0);
            }
        } else try chat_inputs.append(allocator, arg);
    }
    var config = ChatConfig.initFromBuild(config_name);
    if (model_type_) |v| config.model_type = v;
    if (model_name_) |v| config.model_name = v;
    if (num_tokens_) |v| config.num_tokens = v;
    if (chat_format_) |v| config.chat_format = v;
    if (chat_inputs.items.len > 1) config.chat_inputs = chat_inputs.items[1..];
    std.debug.print("\n=== {s} ({s}) ===\n", .{ config.model_type, config.model_name });
    try downloadModel(allocator, "mlx-community", config.model_name);
    var tokenizer = try Tokenizer.init(allocator, config.model_name);
    defer tokenizer.deinit();
    const input_ids = try tokenizer.encodeChat(config.chat_format, config.chat_inputs);
    defer allocator.free(input_ids);
    var transformer = try TransformerUnion.init(allocator, config.model_type, config.model_name);
    defer transformer.deinit();
    const output_ids = try transformer.generate(input_ids, config.num_tokens);
    defer allocator.free(output_ids);
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);
    std.debug.print("\nInput:\n{s}\n\nOutput:\n{s}\n", .{ input_str, output_str });
}

const TransformerUnion = union(enum) {
    llama: LlamaTransformer,
    phi: PhiTransformer,
    qwen: QwenTransformer,

    pub fn init(allocator: std.mem.Allocator, model_type: []const u8, model_name: []const u8) !TransformerUnion {
        if (std.mem.eql(u8, model_type, "llama")) return .{ .llama = try LlamaTransformer.init(allocator, model_name) };
        if (std.mem.eql(u8, model_type, "phi")) return .{ .phi = try PhiTransformer.init(allocator, model_name) };
        if (std.mem.eql(u8, model_type, "qwen")) return .{ .qwen = try QwenTransformer.init(allocator, model_name) };
        return error.UnsupportedModelType;
    }

    pub fn deinit(self: *TransformerUnion) void {
        switch (self.*) {
            inline else => |*t| t.deinit(),
        }
    }

    pub fn generate(self: *TransformerUnion, input: []const u32, num_tokens: usize) ![]u32 {
        return switch (self.*) {
            inline else => |*t| t.generate(input, num_tokens),
        };
    }
};

const ChatConfig = struct {
    model_type: []const u8,
    model_name: []const u8,
    chat_format: ?[]const u8 = null,
    chat_inputs: []const []const u8,
    num_tokens: usize = if (build_options.max) |t| t else 30, // : default

    pub fn initFromBuild(config_name: ?[]const u8) ChatConfig {
        var base_config = ConfigMap.get(if (config_name) |v| v else if (build_options.config) |v| v else "qwq").?; // : default
        if (build_options.format) |v| base_config.chat_format = v;
        if (build_options.max) |v| base_config.num_tokens = v;
        if (build_options.model_type) |v| base_config.model_type = v;
        if (build_options.model_name) |v| base_config.model_name = v;
        return base_config;
    }
};

const ConfigMap = std.StaticStringMap(ChatConfig).initComptime(.{
    .{
        "qwq", @as(ChatConfig, .{
            .model_type = "qwen",
            .model_name = "QwQ-32B-3bit",
            .chat_format =
            \\<|im_start|>user
            \\{s}<|im_end|>
            \\<|im_start|>assistant
            \\<think>
            \\
            ,
            .chat_inputs = &[_][]const u8{"Write a python program to calculate the 10th fibonacci number"},
        }),
    },
    .{
        "r1", @as(ChatConfig, .{
            .model_type = "qwen",
            .model_name = "DeepSeek-R1-Distill-Qwen-32B-4bit",
            .chat_format =
            \\<|im_start|>user
            \\{s}<|im_end|>
            \\<|im_start|>assistant
            \\<think>
            \\
            ,
            .chat_inputs = &[_][]const u8{"Write a python program to calculate the 10th fibonacci number"},
        }),
    },
    .{
        "r1-math", @as(ChatConfig, .{
            .model_type = "qwen",
            .model_name = "DeepSeek-R1-Distill-Qwen-1.5B-4bit",
            .chat_format =
            \\<|im_start|>user
            \\{s}<|im_end|>
            \\<|im_start|>assistant
            \\<think>
            \\
            ,
            .chat_inputs = &[_][]const u8{"Write a python program to calculate the 10th fibonacci number"},
        }),
    },
    .{
        "llama", @as(ChatConfig, .{
            .model_type = "llama",
            .model_name = "Llama-3.2-1B-Instruct-4bit",
            .chat_format =
            \\<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            \\
            \\Cutting Knowledge Date: December 2023
            \\Today Date: 26 Jul 2024
            \\You are a pirate chatbot who always responds in pirate speak!<|eot_id|><|start_header_id|>user<|end_header_id|>
            \\
            \\{s}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            \\
            \\
            ,
            .chat_inputs = &.{"How to get length of ArrayList in Zig?"},
        }),
    },
    .{
        "phi", @as(ChatConfig, .{
            .model_type = "phi",
            .model_name = "phi-4-2bit",
            .chat_format =
            \\<|im_start|>system<|im_sep|>
            \\You are a medieval knight and must provide explanations to modern people.<|im_end|>
            \\<|im_start|>user<|im_sep|>
            \\{s}<|im_end|>
            \\<|im_start|>assistant<|im_sep|>
            \\
            ,
            .chat_inputs = &[_][]const u8{"How should I explain the Internet?"},
        }),
    },
    .{
        "olympic", @as(ChatConfig, .{
            .model_type = "qwen",
            .model_name = "OlympicCoder-7B-4bit",
            .chat_format =
            \\<|im_start|>user
            \\{s}<|im_end|>
            \\<|im_start|>assistant
            \\
            ,
            .chat_inputs = &[_][]const u8{"Write a python program to calculate the 10th fibonacci number"},
        }),
    },
    .{
        "qwen-fim", @as(ChatConfig, .{
            .model_type = "qwen",
            .model_name = "Qwen2.5-Coder-1.5B-4bit",
            .chat_format = "<|fim_prefix|>{s}<|fim_suffix|>{s}<|fim_middle|>",
            .chat_inputs = &.{
                \\def quicksort(arr):
                \\    if len(arr) <= 1:
                \\        return arr
                \\    pivot = arr[len(arr) // 2]
                \\
                ,
                \\    middle = [x for x in arr if x == pivot]
                \\    right = [x for x in arr if x > pivot]
                \\    return quicksort(left) + middle + quicksort(right)
            },
        }),
    },
    .{
        "qwen-coder", @as(ChatConfig, .{
            .model_type = "qwen",
            .model_name = "Qwen2.5-Coder-1.5B-4bit",
            .chat_inputs = &.{
                \\<|repo_name|>library-system
                \\<|file_sep|>library.py
                \\class Book:
                \\    def __init__(self, title, author, isbn, copies):
                \\        self.title = title
                \\        self.author = author
                \\        self.isbn = isbn
                \\        self.copies = copies
                \\
                \\    def __str__(self):
                \\        return f"Title: {self.title}, Author: {self.author}, ISBN: {self.isbn}, Copies: {self.copies}"
                \\
                \\class Library:
                \\    def __init__(self):
                \\        self.books = []
                \\
                \\    def add_book(self, title, author, isbn, copies):
                \\        book = Book(title, author, isbn, copies)
                \\        self.books.append(book)
                \\
                \\    def find_book(self, isbn):
                \\        for book in self.books:
                \\            if book.isbn == isbn:
                \\                return book
                \\        return None
                \\
                \\    def list_books(self):
                \\        return self.books
                \\
                \\<|file_sep|>student.py
                \\class Student:
                \\    def __init__(self, name, id):
                \\        self.name = name
                \\        self.id = id
                \\        self.borrowed_books = []
                \\
                \\    def borrow_book(self, book, library):
                \\        if book and book.copies > 0:
                \\            self.borrowed_books.append(book)
                \\            book.copies -= 1
                \\            return True
                \\        return False
                \\
                \\    def return_book(self, book, library):
                \\        if book in self.borrowed_books:
                \\            self.borrowed_books.remove(book)
                \\            book.copies += 1
                \\            return True
                \\        return False
                \\
                \\<|file_sep|>main.py
                \\from library import Library
                \\from student import Student
                \\
                \\def main():
                \\    # Set up the library with some books
                \\    library = Library()
                \\    library.add_book("The Great Gatsby", "F. Scott Fitzgerald", "1234567890", 3)
                \\    library.add_book("To Kill a Mockingbird", "Harper Lee", "1234567891", 2)
                \\    
                \\    # Set up a student
                \\    student = Student("Alice", "S1")
                \\    
                \\    # Student borrows a book
            },
        }),
    },
});

fn printUsage() void {
    const usage =
        \\Options:
        \\  --config=CONFIG         Config: llama, phi, qwen, olympic, r1, ... (default: qwq)
        \\  --format=FORMAT         Custom chat format template string (defaults based on config)
        \\  --model-type=TYPE       Model type: llama, phi, qwen, ... (defaults based on config)
        \\  --model-name=NAME       Model name (defaults based on config)
        \\  --max=N                 Maximum number of tokens to generate (default: 30)
        \\  --help                  Show this help
        \\
        \\Examples:
        \\  zig build -Dconfig=phi -Dformat={s}
        \\  zig build run-llm -Dmax=100 -- "Write a python function to check if a number is prime"
        \\  zig-out/bin/llm --config=qwen --format={s} "Write a python function to check if a number is prime"
        \\
    ;
    std.debug.print("{s}", .{usage});
}
