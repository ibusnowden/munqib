#!/usr/bin/env python3
"""core.py — minimal agent for pretraining data prep (FineWeb-Edu / ClimbX style)"""

import glob as globlib, json, os, re, subprocess, urllib.request

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1/messages" if OPENROUTER_KEY else "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("MODEL", "anthropic/claude-opus-4.5" if OPENROUTER_KEY else "claude-opus-4-5")

# ANSI colors
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
BLUE, CYAN, GREEN, YELLOW, RED = (
    "\033[34m",
    "\033[36m",
    "\033[32m",
    "\033[33m",
    "\033[31m",
)


# --- Tool implementations ---


def read(args):
    with open(args["path"]) as f:
        lines = f.readlines()
    offset = args.get("offset", 0)
    limit = args.get("limit", len(lines))
    selected = lines[offset : offset + limit]
    return "".join(f"{offset + idx + 1:4}| {line}" for idx, line in enumerate(selected))


def write(args):
    with open(args["path"], "w") as f:
        f.write(args["content"])
    return "ok"


def edit(args):
    with open(args["path"]) as f:
        text = f.read()
    old, new = args["old"], args["new"]
    if old not in text:
        return "error: old_string not found"
    count = text.count(old)
    if not args.get("all") and count > 1:
        return f"error: old_string appears {count} times, must be unique (use all=true)"
    replacement = (
        text.replace(old, new) if args.get("all") else text.replace(old, new, 1)
    )
    with open(args["path"], "w") as f:
        f.write(replacement)
    return "ok"


def glob(args):
    pattern = (args.get("path", ".") + "/" + args["pat"]).replace("//", "/")
    files = globlib.glob(pattern, recursive=True)
    files = sorted(
        files,
        key=lambda f: os.path.getmtime(f) if os.path.isfile(f) else 0,
        reverse=True,
    )
    return "\n".join(files) or "none"


def grep(args):
    pattern = re.compile(args["pat"])
    hits = []
    for filepath in globlib.glob(args.get("path", ".") + "/**", recursive=True):
        try:
            with open(filepath) as f:
                for line_num, line in enumerate(f, 1):
                    if pattern.search(line):
                        hits.append(f"{filepath}:{line_num}:{line.rstrip()}")
        except Exception:
            pass
    return "\n".join(hits[:50]) or "none"


def stats(args):
    path  = args["path"]
    field = args.get("field", "text")
    fmt   = args.get("format", "auto")

    lengths = []
    is_jsonl = None  # determined on first non-empty line if auto

    with open(path) as f:
        if fmt != "text":
            # JSONL path (or auto-detect)
            text_chunks = []
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                if is_jsonl is None:
                    is_jsonl = (fmt == "jsonl") or stripped.startswith("{")
                    if not is_jsonl:
                        text_chunks.append(line)
                        continue
                if is_jsonl:
                    try:
                        doc = json.loads(stripped)
                        lengths.append(len(doc.get(field) or ""))
                    except Exception:
                        pass
                else:
                    text_chunks.append(line)
            if is_jsonl is False:
                # Fell through to text mode mid-file — finish collecting
                content = "".join(text_chunks)
                sep = "<|endoftext|>" if "<|endoftext|>" in content else "\n\n"
                for doc in content.split(sep):
                    doc = doc.strip()
                    if doc:
                        lengths.append(len(doc))
        else:
            is_jsonl = False
            content = f.read()
            sep = "<|endoftext|>" if "<|endoftext|>" in content else "\n\n"
            for doc in content.split(sep):
                doc = doc.strip()
                if doc:
                    lengths.append(len(doc))

    if is_jsonl is None:
        is_jsonl = False  # empty file

    if not lengths:
        return "docs: 0\nchars: 0\ntokens: ~0  (chars / 4)\nlengths: no docs found"

    count       = len(lengths)
    total_chars = sum(lengths)
    tokens      = total_chars // 4
    mean_len    = total_chars // count
    sorted_lens = sorted(lengths)
    p50 = sorted_lens[len(sorted_lens) // 2]
    p90 = sorted_lens[len(sorted_lens) * 9 // 10]

    def fmt(n): return f"{n:,}"
    fmt_name  = "jsonl" if is_jsonl else "text"
    field_str = f"  field: {field}" if is_jsonl else ""
    return (
        f"docs:    {fmt(count)}\n"
        f"chars:   {fmt(total_chars)}\n"
        f"tokens:  ~{fmt(tokens)}  (chars / 4)\n"
        f"lengths: min={fmt(min(lengths))}  mean={fmt(mean_len)}  p50={fmt(p50)}  p90={fmt(p90)}  max={fmt(max(lengths))}\n"
        f"format:  {fmt_name}{field_str}"
    )


def bash(args):
    proc = subprocess.Popen(
        args["cmd"], shell=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )
    output_lines = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(f"  {DIM}│ {line.rstrip()}{RESET}", flush=True)
                output_lines.append(line)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        output_lines.append("\n(timed out after 30s)")
    return "".join(output_lines).strip() or "(empty)"


# --- Tool definitions: (description, schema, function) ---

TOOLS = {
    "read": (
        "Read file with line numbers (file path, not directory)",
        {"path": "string", "offset": "number?", "limit": "number?"},
        read,
    ),
    "write": (
        "Write content to file",
        {"path": "string", "content": "string"},
        write,
    ),
    "edit": (
        "Replace old with new in file (old must be unique unless all=true)",
        {"path": "string", "old": "string", "new": "string", "all": "boolean?"},
        edit,
    ),
    "glob": (
        "Find files by pattern, sorted by mtime",
        {"pat": "string", "path": "string?"},
        glob,
    ),
    "grep": (
        "Search files for regex pattern",
        {"pat": "string", "path": "string?"},
        grep,
    ),
    "bash": (
        "Run shell command",
        {"cmd": "string"},
        bash,
    ),
    "stats": (
        "Corpus statistics for a JSONL or plain-text file (doc count, chars, tokens, length distribution)",
        {"path": "string", "format": "string?", "field": "string?"},
        stats,
    ),
}


def run_tool(name, args):
    try:
        return TOOLS[name][2](args)
    except Exception as err:
        return f"error: {err}"


def make_schema():
    result = []
    for name, (description, params, _fn) in TOOLS.items():
        properties = {}
        required = []
        for param_name, param_type in params.items():
            is_optional = param_type.endswith("?")
            base_type = param_type.rstrip("?")
            properties[param_name] = {
                "type": "integer" if base_type == "number" else base_type
            }
            if not is_optional:
                required.append(param_name)
        result.append(
            {
                "name": name,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        )
    return result


def call_api(messages, system_prompt):
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(
            {
                "model": MODEL,
                "max_tokens": 8192,
                "system": system_prompt,
                "messages": messages,
                "tools": make_schema(),
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
            **({"Authorization": f"Bearer {OPENROUTER_KEY}"} if OPENROUTER_KEY else {"x-api-key": os.environ.get("ANTHROPIC_API_KEY", "")}),
        },
    )
    response = urllib.request.urlopen(request)
    return json.loads(response.read())


def separator():
    return f"{DIM}{'─' * min(os.get_terminal_size(fallback=(80, 24)).columns, 80)}{RESET}"


def render_markdown(text):
    return re.sub(r"\*\*(.+?)\*\*", f"{BOLD}\\1{RESET}", text)


def main():
    print(f"{BOLD}core{RESET} | {DIM}{MODEL} ({'OpenRouter' if OPENROUTER_KEY else 'Anthropic'}) | {os.getcwd()}{RESET}\n")
    messages = []
    dp_ctx = " dataprep.py available: filter/dedup/score/sample subcommands for pretraining data." if os.path.exists("dataprep.py") else ""
    system_prompt = f"Agent for pretraining data prep (FineWeb-Edu / ClimbX style). cwd: {os.getcwd()}.{dp_ctx}"

    while True:
        try:
            print(separator())
            user_input = input(f"{BOLD}{BLUE}❯{RESET} ").strip()
            print(separator())
            if not user_input:
                continue
            if user_input in ("/q", "exit"):
                break
            if user_input == "/c":
                messages = []
                print(f"{GREEN}⏺ Cleared conversation{RESET}")
                continue

            messages.append({"role": "user", "content": user_input})

            # agentic loop: keep calling API until no more tool calls
            while True:
                response = call_api(messages, system_prompt)
                content_blocks = response.get("content", [])
                tool_results = []

                for block in content_blocks:
                    if block["type"] == "text":
                        print(f"\n{CYAN}⏺{RESET} {render_markdown(block['text'])}")

                    if block["type"] == "tool_use":
                        tool_name = block["name"]
                        tool_args = block["input"]
                        arg_preview = str(next(iter(tool_args.values()), ""))[:50]
                        print(
                            f"\n{GREEN}⏺ {tool_name.capitalize()}{RESET}({DIM}{arg_preview}{RESET})"
                        )

                        result = run_tool(tool_name, tool_args)
                        result_lines = result.split("\n")
                        preview = result_lines[0][:60]
                        if len(result_lines) > 1:
                            preview += f" ... +{len(result_lines) - 1} lines"
                        elif len(result_lines[0]) > 60:
                            preview += "..."
                        print(f"  {DIM}⎿  {preview}{RESET}")

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block["id"],
                                "content": result,
                            }
                        )

                messages.append({"role": "assistant", "content": content_blocks})

                if not tool_results:
                    break
                messages.append({"role": "user", "content": tool_results})

            print()

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as err:
            print(f"{RED}⏺ Error: {err}{RESET}")


if __name__ == "__main__":
    main()
