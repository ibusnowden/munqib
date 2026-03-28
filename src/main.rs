use std::io::{self, BufRead, Write};
use std::process::Command;
use std::{env, fs};

// ── ANSI constants ──────────────────────────────────────────────────────────
const RESET: &str = "\x1b[0m";
const BOLD:  &str = "\x1b[1m";
const DIM:   &str = "\x1b[2m";
const BLUE:  &str = "\x1b[34m";
const CYAN:  &str = "\x1b[36m";
const GREEN: &str = "\x1b[32m";
const RED:   &str = "\x1b[31m";

// ── Provider ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Provider {
    Anthropic,
    OpenRouter,
    OpenAI,
    Gemini,
}

fn detect_provider() -> (Provider, String) {
    if let Ok(k) = env::var("OPENROUTER_API_KEY") { return (Provider::OpenRouter, k); }
    if let Ok(k) = env::var("ANTHROPIC_API_KEY")  { return (Provider::Anthropic,  k); }
    if let Ok(k) = env::var("OPENAI_API_KEY")     { return (Provider::OpenAI,     k); }
    let gemini = env::var("GEMINI_API_KEY").or_else(|_| env::var("GOOGLE_API_KEY"));
    if let Ok(k) = gemini { return (Provider::Gemini, k); }
    (Provider::Anthropic, String::new())  // will fail at API with a clear error
}

fn default_model(provider: &Provider) -> &'static str {
    match provider {
        Provider::Anthropic  => "claude-opus-4-5",
        Provider::OpenRouter => "anthropic/claude-opus-4-5",
        Provider::OpenAI     => "gpt-4o",
        Provider::Gemini     => "gemini-2.0-flash",
    }
}

fn provider_name(provider: &Provider) -> &'static str {
    match provider {
        Provider::Anthropic  => "Anthropic",
        Provider::OpenRouter => "OpenRouter",
        Provider::OpenAI     => "OpenAI",
        Provider::Gemini     => "Gemini",
    }
}

// ── Tool definitions ─────────────────────────────────────────────────────────

struct ToolDef {
    name:        &'static str,
    description: &'static str,
    /// (param_name, json_type, required)
    params:      &'static [(&'static str, &'static str, bool)],
}

const TOOLS: &[ToolDef] = &[
    ToolDef {
        name: "bash",
        description: "Run a shell command (stdout+stderr merged)",
        params: &[("cmd", "string", true)],
    },
    ToolDef {
        name: "read",
        description: "Read a file with line numbers",
        params: &[
            ("path",   "string",  true),
            ("offset", "integer", false),
            ("limit",  "integer", false),
        ],
    },
    ToolDef {
        name: "write",
        description: "Write content to a file (overwrites)",
        params: &[
            ("path",    "string", true),
            ("content", "string", true),
        ],
    },
    ToolDef {
        name: "stats",
        description: "Corpus statistics for a JSONL or plain-text file (doc count, chars, estimated tokens, length distribution)",
        params: &[
            ("path",   "string", true),
            ("format", "string", false),  // "jsonl" | "text" — auto-detected if omitted
            ("field",  "string", false),  // JSON field for JSONL, default "text"
        ],
    },
];

// ── Tool implementations ──────────────────────────────────────────────────────

fn tool_bash(args: &serde_json::Value) -> String {
    let cmd = match args["cmd"].as_str() {
        Some(s) => s,
        None => return "error: missing cmd".into(),
    };
    match Command::new("sh").arg("-c").arg(format!("{} 2>&1", cmd)).output() {
        Ok(o) => {
            let text = String::from_utf8_lossy(&o.stdout);
            for line in text.lines() {
                println!("  {}│ {}{}", DIM, line, RESET);
            }
            let trimmed = text.trim().to_string();
            if trimmed.is_empty() { "(empty)".into() } else { trimmed }
        }
        Err(e) => format!("error: {}", e),
    }
}

fn tool_read(args: &serde_json::Value) -> String {
    let path = match args["path"].as_str() {
        Some(s) => s,
        None => return "error: missing path".into(),
    };
    let content = match fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => return format!("error: {}", e),
    };
    let lines: Vec<&str> = content.lines().collect();
    let total  = lines.len();
    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
    let limit  = args["limit"].as_u64().unwrap_or(total as u64) as usize;
    let end    = (offset + limit).min(total);
    lines[offset..end]
        .iter()
        .enumerate()
        .map(|(i, line)| format!("{:4}| {}", offset + i + 1, line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn tool_write(args: &serde_json::Value) -> String {
    let path    = match args["path"].as_str()    { Some(s) => s, None => return "error: missing path".into() };
    let content = match args["content"].as_str() { Some(s) => s, None => return "error: missing content".into() };
    match fs::write(path, content) {
        Ok(_)  => "ok".into(),
        Err(e) => format!("error: {}", e),
    }
}

fn tool_stats(args: &serde_json::Value) -> String {
    use std::io::BufReader;

    let path = match args["path"].as_str() {
        Some(s) => s,
        None => return "error: missing path".into(),
    };
    let field = args["field"].as_str().unwrap_or("text");

    let file = match fs::File::open(path) {
        Ok(f)  => f,
        Err(e) => return format!("error: {}", e),
    };
    let reader = BufReader::new(file);

    // Auto-detect format from first non-empty line
    let mut lines_iter = reader.lines();
    let mut first_line = String::new();
    for line in lines_iter.by_ref() {
        match line {
            Ok(l) if !l.trim().is_empty() => { first_line = l; break; }
            Ok(_) => continue,
            Err(e) => return format!("error reading file: {}", e),
        }
    }

    let fmt_arg = args["format"].as_str().unwrap_or("");
    let is_jsonl = if fmt_arg == "jsonl" {
        true
    } else if fmt_arg == "text" {
        false
    } else {
        first_line.trim_start().starts_with('{')
    };

    let mut lengths: Vec<usize> = Vec::new();

    let process_line = |line: &str, lengths: &mut Vec<usize>| {
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = v[field].as_str() {
                lengths.push(text.len());
            }
        }
    };

    if is_jsonl {
        // Process first line
        if !first_line.is_empty() {
            process_line(&first_line, &mut lengths);
        }
        for line in lines_iter {
            match line {
                Ok(l) if !l.trim().is_empty() => process_line(&l, &mut lengths),
                Ok(_)  => continue,
                Err(e) => return format!("error reading file: {}", e),
            }
        }
    } else {
        // Text mode: collect all content, split on <|endoftext|> or \n\n
        let mut all_content = first_line;
        all_content.push('\n');
        for line in lines_iter {
            match line {
                Ok(l)  => { all_content.push_str(&l); all_content.push('\n'); }
                Err(e) => return format!("error reading file: {}", e),
            }
        }
        let docs: Vec<&str> = if all_content.contains("<|endoftext|>") {
            all_content.split("<|endoftext|>").collect()
        } else {
            all_content.split("\n\n").collect()
        };
        for doc in docs {
            let trimmed = doc.trim();
            if !trimmed.is_empty() {
                lengths.push(trimmed.len());
            }
        }
    }

    if lengths.is_empty() {
        return "docs: 0\nchars: 0\ntokens: ~0  (chars / 4)\nlengths: no docs found".into();
    }

    let count      = lengths.len();
    let total_chars: usize = lengths.iter().sum();
    let min_len    = *lengths.iter().min().unwrap();
    let max_len    = *lengths.iter().max().unwrap();
    let mean_len   = total_chars / count;
    let tokens     = total_chars / 4;

    lengths.sort_unstable();
    let p50 = lengths[lengths.len() / 2];
    let p90 = lengths[lengths.len() * 9 / 10];

    let fmt_name = if is_jsonl { "jsonl" } else { "text" };
    let field_info = if is_jsonl { format!("  field: {}", field) } else { String::new() };

    format!(
        "docs:    {}\nchars:   {}\ntokens:  ~{}  (chars / 4)\nlengths: min={}  mean={}  p50={}  p90={}  max={}\nformat:  {}{}",
        fmt_num(count),
        fmt_num(total_chars),
        fmt_num(tokens),
        fmt_num(min_len),
        fmt_num(mean_len),
        fmt_num(p50),
        fmt_num(p90),
        fmt_num(max_len),
        fmt_name,
        field_info,
    )
}

fn fmt_num(n: usize) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::new();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 { out.push(','); }
        out.push(b as char);
    }
    out
}

fn run_tool(name: &str, args: &serde_json::Value) -> String {
    match name {
        "bash"  => tool_bash(args),
        "read"  => tool_read(args),
        "write" => tool_write(args),
        "stats" => tool_stats(args),
        other   => format!("error: unknown tool '{}'", other),
    }
}

// ── Schema builders ───────────────────────────────────────────────────────────

/// Anthropic format: { name, description, input_schema: { type, properties, required } }
fn build_anthropic_tools_schema() -> serde_json::Value {
    serde_json::Value::Array(TOOLS.iter().map(|t| {
        let mut props = serde_json::Map::new();
        let mut req   = Vec::new();
        for (pname, ptype, required) in t.params {
            props.insert(pname.to_string(), serde_json::json!({ "type": ptype }));
            if *required { req.push(serde_json::Value::String(pname.to_string())); }
        }
        serde_json::json!({
            "name": t.name,
            "description": t.description,
            "input_schema": { "type": "object", "properties": props, "required": req }
        })
    }).collect())
}

/// OpenAI format: { type: "function", function: { name, description, parameters: { … } } }
fn build_openai_tools_schema() -> serde_json::Value {
    serde_json::Value::Array(TOOLS.iter().map(|t| {
        let mut props = serde_json::Map::new();
        let mut req   = Vec::new();
        for (pname, ptype, required) in t.params {
            props.insert(pname.to_string(), serde_json::json!({ "type": ptype }));
            if *required { req.push(serde_json::Value::String(pname.to_string())); }
        }
        serde_json::json!({
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": { "type": "object", "properties": props, "required": req }
            }
        })
    }).collect())
}

// ── Message types ─────────────────────────────────────────────────────────────

struct ToolCall {
    id:   String,
    name: String,
    args: serde_json::Value,
}

enum MsgContent {
    Text(String),
    ToolCalls(Vec<ToolCall>),
    ToolResults(Vec<(String, String, String)>),  // (id, name, result)
}

struct Message {
    role:    String,
    content: MsgContent,
}

// ── Message serializers ───────────────────────────────────────────────────────

fn serialize_anthropic(messages: &[Message]) -> Vec<serde_json::Value> {
    messages.iter().map(|m| {
        let content = match &m.content {
            MsgContent::Text(t) => serde_json::Value::String(t.clone()),
            MsgContent::ToolCalls(calls) => serde_json::Value::Array(calls.iter().map(|c| {
                serde_json::json!({
                    "type": "tool_use",
                    "id":   c.id,
                    "name": c.name,
                    "input": c.args
                })
            }).collect()),
            MsgContent::ToolResults(results) => serde_json::Value::Array(results.iter().map(|(id, _name, result)| {
                serde_json::json!({
                    "type":        "tool_result",
                    "tool_use_id": id,
                    "content":     result
                })
            }).collect()),
        };
        serde_json::json!({ "role": m.role, "content": content })
    }).collect()
}

fn serialize_openai(messages: &[Message], system_prompt: &str) -> Vec<serde_json::Value> {
    let mut out = vec![serde_json::json!({ "role": "system", "content": system_prompt })];
    for m in messages {
        match &m.content {
            MsgContent::Text(t) => {
                out.push(serde_json::json!({ "role": m.role, "content": t }));
            }
            MsgContent::ToolCalls(calls) => {
                let tool_calls: Vec<serde_json::Value> = calls.iter().map(|c| {
                    serde_json::json!({
                        "id":   c.id,
                        "type": "function",
                        "function": {
                            "name":      c.name,
                            "arguments": serde_json::to_string(&c.args).unwrap_or_default()
                        }
                    })
                }).collect();
                out.push(serde_json::json!({
                    "role":       "assistant",
                    "content":    null,
                    "tool_calls": tool_calls
                }));
            }
            MsgContent::ToolResults(results) => {
                // OpenAI requires one message per tool result
                for (id, _name, result) in results {
                    out.push(serde_json::json!({
                        "role":         "tool",
                        "tool_call_id": id,
                        "content":      result
                    }));
                }
            }
        }
    }
    out
}

// ── Response parsers ──────────────────────────────────────────────────────────

fn parse_anthropic_response(resp: &serde_json::Value) -> (Option<String>, Vec<ToolCall>) {
    let mut text  = None;
    let mut calls = Vec::new();
    if let Some(blocks) = resp["content"].as_array() {
        for block in blocks {
            match block["type"].as_str() {
                Some("text") => {
                    text = block["text"].as_str().map(|s| s.to_string());
                }
                Some("tool_use") => {
                    calls.push(ToolCall {
                        id:   block["id"].as_str().unwrap_or("").to_string(),
                        name: block["name"].as_str().unwrap_or("").to_string(),
                        args: block["input"].clone(),
                    });
                }
                _ => {}
            }
        }
    }
    (text, calls)
}

fn parse_openai_response(resp: &serde_json::Value) -> (Option<String>, Vec<ToolCall>) {
    let mut text  = None;
    let mut calls = Vec::new();
    if let Some(msg) = resp["choices"][0]["message"].as_object() {
        if let Some(t) = msg.get("content").and_then(|v| v.as_str()) {
            if !t.is_empty() { text = Some(t.to_string()); }
        }
        if let Some(tcs) = msg.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let args_str = tc["function"]["arguments"].as_str().unwrap_or("{}");
                let args: serde_json::Value = serde_json::from_str(args_str).unwrap_or_default();
                calls.push(ToolCall {
                    id:   tc["id"].as_str().unwrap_or("").to_string(),
                    name: tc["function"]["name"].as_str().unwrap_or("").to_string(),
                    args,
                });
            }
        }
    }
    (text, calls)
}

// ── API call ──────────────────────────────────────────────────────────────────

fn call_api(
    messages: &[Message],
    system_prompt: &str,
    provider: &Provider,
    api_key: &str,
    model: &str,
) -> Result<serde_json::Value, String> {
    let is_openai_compat = matches!(provider, Provider::OpenAI | Provider::Gemini);

    let url = match provider {
        Provider::Anthropic  => "https://api.anthropic.com/v1/messages",
        Provider::OpenRouter => "https://openrouter.ai/api/v1/messages",
        Provider::OpenAI     => "https://api.openai.com/v1/chat/completions",
        Provider::Gemini     => "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
    };

    let (auth_header, auth_value) = match provider {
        Provider::Anthropic => ("x-api-key", api_key.to_string()),
        _                   => ("Authorization", format!("Bearer {}", api_key)),
    };

    let body = if is_openai_compat {
        serde_json::json!({
            "model":    model,
            "messages": serialize_openai(messages, system_prompt),
            "tools":    build_openai_tools_schema()
        })
    } else {
        serde_json::json!({
            "model":      model,
            "max_tokens": 8192,
            "system":     system_prompt,
            "messages":   serialize_anthropic(messages),
            "tools":      build_anthropic_tools_schema()
        })
    };

    let mut req = ureq::post(url)
        .set("Content-Type", "application/json")
        .set(auth_header, &auth_value);
    if !is_openai_compat {
        req = req.set("anthropic-version", "2023-06-01");
    }

    req.send_json(body)
        .map_err(|e| format!("{}", e))?
        .into_json::<serde_json::Value>()
        .map_err(|e| format!("{}", e))
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn print_separator() {
    let width: usize = env::var("COLUMNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(80);
    println!("{}{}{}", DIM, "─".repeat(width.min(80)), RESET);
}

/// Replace **text** with ANSI bold — no regex.
fn render_bold(text: &str) -> String {
    let mut out   = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '*' && chars.peek() == Some(&'*') {
            chars.next();
            let mut inner  = String::new();
            let mut closed = false;
            loop {
                match chars.next() {
                    Some('*') if chars.peek() == Some(&'*') => { chars.next(); closed = true; break; }
                    Some(ch) => inner.push(ch),
                    None     => break,
                }
            }
            if closed { out.push_str(BOLD); out.push_str(&inner); out.push_str(RESET); }
            else      { out.push_str("**"); out.push_str(&inner); }
        } else {
            out.push(c);
        }
    }
    out
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None    => String::new(),
        Some(f) => f.to_uppercase().to_string() + c.as_str(),
    }
}

fn first_arg_preview(args: &serde_json::Value) -> String {
    if let Some(obj) = args.as_object() {
        if let Some(v) = obj.values().next() {
            let s = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            return s.chars().take(50).collect();
        }
    }
    String::new()
}

fn print_tool_result_preview(result: &str) {
    let lines: Vec<&str> = result.lines().collect();
    let first = lines.first().copied().unwrap_or("");
    let preview: String = first.chars().take(60).collect();
    let suffix = if lines.len() > 1 {
        format!(" ... +{} lines", lines.len() - 1)
    } else if first.len() > 60 {
        "...".into()
    } else {
        String::new()
    };
    println!("  {}⎿  {}{}{}", DIM, preview, suffix, RESET);
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let (provider, api_key) = detect_provider();
    let model = env::var("MODEL").unwrap_or_else(|_| default_model(&provider).to_string());
    let cwd   = env::current_dir().unwrap_or_default();

    println!("{}nanocode{} | {}{} ({}) | {}{}\n",
        BOLD, RESET, DIM, model, provider_name(&provider), cwd.display(), RESET);

    let dp_ctx = if std::path::Path::new("dataprep.py").exists() {
        " dataprep.py available: filter/dedup/score/sample subcommands for pretraining data."
    } else { "" };
    let system_prompt = format!("Concise coding assistant. cwd: {}.{}", cwd.display(), dp_ctx);
    let mut messages: Vec<Message> = Vec::new();

    let stdin = io::stdin();
    loop {
        print_separator();
        print!("{}{}>{}  ", BOLD, BLUE, RESET);
        io::stdout().flush().unwrap();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) | Err(_) => break,
            Ok(_) => {}
        }
        let input = line.trim().to_string();
        print_separator();

        if input.is_empty() { continue; }
        if input == "/q" || input == "exit" { break; }
        if input == "/c" {
            messages.clear();
            println!("{}⏺ Cleared conversation{}", GREEN, RESET);
            continue;
        }

        messages.push(Message { role: "user".into(), content: MsgContent::Text(input) });

        // agentic loop
        loop {
            let response = match call_api(&messages, &system_prompt, &provider, &api_key, &model) {
                Ok(r)  => r,
                Err(e) => { println!("{}⏺ Error: {}{}", RED, e, RESET); break; }
            };

            let is_openai_compat = matches!(provider, Provider::OpenAI | Provider::Gemini);
            let (text, tool_calls) = if is_openai_compat {
                parse_openai_response(&response)
            } else {
                parse_anthropic_response(&response)
            };

            if let Some(t) = &text {
                println!("\n{}⏺{}  {}", CYAN, RESET, render_bold(t));
            }

            let mut results: Vec<(String, String, String)> = Vec::new();
            for tc in &tool_calls {
                let preview = first_arg_preview(&tc.args);
                println!("\n{}⏺ {}{}({}{}{}){}",
                    GREEN, capitalize(&tc.name), RESET,
                    DIM, preview, RESET, RESET);
                let result = run_tool(&tc.name, &tc.args);
                print_tool_result_preview(&result);
                results.push((tc.id.clone(), tc.name.clone(), result));
            }

            // Store assistant turn
            if tool_calls.is_empty() {
                if let Some(t) = text {
                    messages.push(Message { role: "assistant".into(), content: MsgContent::Text(t) });
                }
                break;
            }
            messages.push(Message {
                role:    "assistant".into(),
                content: MsgContent::ToolCalls(
                    tool_calls.into_iter().map(|tc| ToolCall { id: tc.id, name: tc.name, args: tc.args }).collect()
                ),
            });
            messages.push(Message { role: "user".into(), content: MsgContent::ToolResults(results) });
        }

        println!();
    }
}
