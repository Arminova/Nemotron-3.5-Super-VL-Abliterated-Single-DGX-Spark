#!/bin/bash
# tools/check.sh: three requests against the running server (./start.sh), each must come back right: a sum (text),
# a weather question (a tool call) and a red square (an image, PNG). Greedy, thinking off. Settings: config.sh.
set -euo pipefail
D=$(cd "$(dirname "$0")/.." && pwd)
. "$D/config.sh"
host=$BIND
[ "$host" = 0.0.0.0 ] && host=127.0.0.1
export CHECK_URL="http://$host:$PORT/v1/chat/completions" CHECK_MODEL="$SERVED_MODEL_NAME" CHECK_KEY="${API_KEY:-}"
exec python3 - << 'EOF'
import base64, json, os, struct, sys, urllib.request, zlib

def png(w, h, rgb):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))

def ask(content, **extra):
    body = {"model": os.environ["CHECK_MODEL"], "messages": [{"role": "user", "content": content}],
            "max_tokens": 64, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}, **extra}
    headers = {"Content-Type": "application/json"}
    if os.environ["CHECK_KEY"]:
        headers["Authorization"] = "Bearer " + os.environ["CHECK_KEY"]
    req = urllib.request.Request(os.environ["CHECK_URL"], json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]

weather = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
red = "data:image/png;base64," + base64.b64encode(png(64, 64, (220, 20, 20))).decode()
checks = [
    ("text", lambda: ask("What is 17 * 23? Reply with the number only."),
     lambda m: "391" in (m.get("content") or "")),
    ("tool call", lambda: ask("What's the weather in Paris right now?", tools=[weather]),
     lambda m: any(c["function"]["name"] == "get_weather" and "paris" in c["function"]["arguments"].lower()
                   for c in m.get("tool_calls") or [])),
    ("image", lambda: ask([{"type": "image_url", "image_url": {"url": red}},
                           {"type": "text", "text": "What colour is this image? Reply with one word."}]),
     lambda m: "red" in (m.get("content") or "").lower()),
]
bad = 0
for name, run, ok in checks:
    try:
        m = run()
        good = ok(m)
        shown = m.get("content") or json.dumps(m.get("tool_calls"))
    except Exception as e:
        good, shown = False, repr(e)
    bad += not good
    print(("\033[1;32m✓\033[0m " if good else "\033[1;31m✗\033[0m ") + f"{name}: {shown.strip()[:120]}")
sys.exit(1 if bad else 0)
EOF
