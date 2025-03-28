//! main.zig - LLM Chat Interface
//!
//! Entry point for the Llama language model chat application. Manages the
//! model download process, initializes tokenizer and transformer components,
//! handles prompt formatting, token generation, and text decoding. Provides
//! an interactive chat interface to the model for text generation.
//!
//! Copyright 2025 Joe

const std = @import("std");
const download = @import("utils.zig").download;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Transformer = @import("transformer.zig").DefaultTransformer;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    defer _ = gpa.deinit();

    const model_name = "Llama-3.2-1B-Instruct-4bit";
    const sys_prompt = "You are a helpful assistant.";
    const num_tokens = 100;

    // Download model files
    try download(allocator, model_name);

    // Load tokenizer
    var tokenizer = try Tokenizer.init(allocator, model_name);
    defer tokenizer.deinit();

    // Load transformer
    var transformer = try Transformer.init(allocator, model_name);
    defer transformer.deinit();

    // Prompt user for input
    const stdin = std.io.getStdIn().reader();
    std.debug.print("\n\n===============\n\nEnter your message: ", .{});
    var input_buffer: [1024]u8 = undefined;
    const input_slice = try stdin.readUntilDelimiterOrEof(&input_buffer, '\n') orelse "";
    const user_input = try allocator.dupe(u8, input_slice);
    defer allocator.free(user_input);

    // Encode input string to token IDs (chat format)
    const input_ids = try tokenizer.encodeChat(null, sys_prompt, user_input);
    defer allocator.free(input_ids);

    // Generate new token IDs
    const output_ids = try transformer.generate(input_ids, num_tokens);
    defer allocator.free(output_ids);

    // Decode input and output token IDs
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);

    std.debug.print("\nInput: {s}Output: {s}\n", .{ input_str, output_str });
}

test "Llama-3.2-1B-Instruct-4bit chat" {
    std.debug.print("\n=== MAIN.ZIG ===\n\n", .{});
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    const model_name = "Llama-3.2-1B-Instruct-4bit";
    const sys_prompt = "You are a helpful assistant.";
    const user_input = "Hello world";
    const num_tokens = 10;
    try download(allocator, model_name);
    var tokenizer = try Tokenizer.init(allocator, model_name);
    defer tokenizer.deinit();
    var transformer = try Transformer.init(allocator, model_name);
    defer transformer.deinit();
    const input_ids = try tokenizer.encodeChat(null, sys_prompt, user_input);
    defer allocator.free(input_ids);
    const output_ids = try transformer.generate(input_ids, num_tokens);
    defer allocator.free(output_ids);
    const input_str = try tokenizer.decode(input_ids);
    defer allocator.free(input_str);
    const output_str = try tokenizer.decode(output_ids);
    defer allocator.free(output_str);
    std.debug.print("\nInput: {s}Output: {s}\n", .{ input_str, output_str });
}
