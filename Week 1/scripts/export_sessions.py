"""Export all Claude Code sessions for this project to readable markdown files."""
import json, os, sys, glob
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

SESSIONS_DIR = r'C:\Users\nguye\.claude\projects\d--Quant-Finance-Quant-Program-Week-1'
OUTPUT_DIR = r'd:\Quant Finance\Quant Program\Week 1\antigravity_export'

def parse_timestamp(ts):
    """Parse ISO timestamp or epoch ms to readable string."""
    if isinstance(ts, (int, float)):
        return datetime.utcfromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')
    if isinstance(ts, str):
        # ISO format
        try:
            return ts[:19].replace('T', ' ') + ' UTC'
        except:
            return str(ts)
    return str(ts)

def extract_text_from_content(content):
    """Extract readable text from message content (can be string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'text':
                    parts.append(block.get('text', ''))
                elif block.get('type') == 'tool_use':
                    tool_name = block.get('name', 'unknown_tool')
                    tool_input = block.get('input', {})
                    # Summarize tool calls
                    if tool_name == 'Read':
                        parts.append(f"[Tool: Read file `{tool_input.get('file_path', '?')}`]")
                    elif tool_name == 'Write':
                        fp = tool_input.get('file_path', '?')
                        parts.append(f"[Tool: Write file `{fp}`]")
                    elif tool_name == 'Edit':
                        fp = tool_input.get('file_path', '?')
                        parts.append(f"[Tool: Edit file `{fp}`]")
                    elif tool_name == 'Bash':
                        cmd = tool_input.get('command', '?')
                        desc = tool_input.get('description', '')
                        if len(cmd) > 300:
                            cmd = cmd[:300] + '...'
                        parts.append(f"[Tool: Bash] `{desc or cmd}`")
                    elif tool_name == 'Grep':
                        pattern = tool_input.get('pattern', '?')
                        parts.append(f"[Tool: Grep `{pattern}`]")
                    elif tool_name == 'Glob':
                        pattern = tool_input.get('pattern', '?')
                        parts.append(f"[Tool: Glob `{pattern}`]")
                    elif tool_name == 'Agent':
                        desc = tool_input.get('description', tool_input.get('prompt', '?'))[:200]
                        parts.append(f"[Tool: Agent — {desc}]")
                    else:
                        parts.append(f"[Tool: {tool_name}]")
                elif block.get('type') == 'tool_result':
                    # Skip tool results (too verbose)
                    result_content = block.get('content', '')
                    if isinstance(result_content, str) and len(result_content) > 500:
                        result_content = result_content[:500] + '...'
                    elif isinstance(result_content, list):
                        result_content = '[complex result]'
                    parts.append(f"[Tool Result: {str(result_content)[:200]}]")
                elif block.get('type') == 'image':
                    parts.append("[Image]")
            elif isinstance(block, str):
                parts.append(block)
        return '\n'.join(parts)
    return str(content)

def process_session(filepath):
    """Parse a .jsonl session file and return structured messages."""
    events = []
    with open(filepath, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError:
                pass
    return events

def events_to_markdown(events, session_id):
    """Convert events to a readable markdown document."""
    lines = []

    # Get session metadata from first event
    first_ts = events[0].get('timestamp', '?') if events else '?'

    # Count messages by type
    user_count = sum(1 for e in events if e.get('type') == 'user')
    assistant_count = sum(1 for e in events if e.get('type') == 'assistant')

    lines.append(f"# Claude Code Session: `{session_id[:8]}...`")
    lines.append(f"")
    lines.append(f"- **Session ID**: `{session_id}`")
    lines.append(f"- **Started**: {parse_timestamp(first_ts)}")
    lines.append(f"- **Total events**: {len(events)}")
    lines.append(f"- **User messages**: {user_count}")
    lines.append(f"- **Assistant messages**: {assistant_count}")
    lines.append(f"- **Workspace**: `d:\\Quant Finance\\Quant Program\\Week 1`")
    lines.append(f"- **Exported**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    msg_num = 0
    for event in events:
        event_type = event.get('type', '')
        ts = event.get('timestamp', '')
        ts_str = parse_timestamp(ts) if ts else ''

        if event_type == 'user':
            msg_num += 1
            msg = event.get('message', {})
            content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
            text = extract_text_from_content(content)

            # Clean up IDE noise
            text = text.strip()
            if not text:
                continue

            lines.append(f"## User [{msg_num}]  `{ts_str}`")
            lines.append(f"")
            lines.append(text)
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

        elif event_type == 'assistant':
            msg = event.get('message', {})
            content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
            text = extract_text_from_content(content)

            text = text.strip()
            if not text:
                continue

            # Truncate very long assistant responses (e.g., full file writes)
            if len(text) > 5000:
                text = text[:5000] + f"\n\n... [truncated, {len(text)} chars total]"

            lines.append(f"## Assistant  `{ts_str}`")
            lines.append(f"")
            lines.append(text)
            lines.append(f"")
            lines.append(f"---")
            lines.append(f"")

    return '\n'.join(lines)

# Process all sessions
session_files = sorted(glob.glob(os.path.join(SESSIONS_DIR, '*.jsonl')))

# Sort by timestamp
session_info = []
for f in session_files:
    events = process_session(f)
    if events:
        ts = events[0].get('timestamp', '')
        sid = os.path.basename(f).replace('.jsonl', '')
        size = os.path.getsize(f)
        session_info.append((ts, sid, f, events, size))

session_info.sort(key=lambda x: str(x[0]))

print(f"Found {len(session_info)} sessions\n")

for i, (ts, sid, filepath, events, size) in enumerate(session_info, 1):
    # Determine session name based on content
    user_msgs = []
    for e in events:
        if e.get('type') == 'user':
            msg = e.get('message', {})
            content = msg.get('content', '') if isinstance(msg, dict) else ''
            text = extract_text_from_content(content)
            if text.strip():
                user_msgs.append(text.strip())

    # Generate filename
    ts_str = parse_timestamp(ts).replace(' UTC', '').replace(' ', '_').replace(':', '')
    out_name = f"claude_code_session_{i:02d}_{sid[:8]}.md"
    out_path = os.path.join(OUTPUT_DIR, out_name)

    md = events_to_markdown(events, sid)

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)

    out_size = os.path.getsize(out_path)
    first_user = user_msgs[0][:100].replace('\n', ' ') if user_msgs else '(no user messages)'
    print(f"Session {i}: {sid[:8]}...")
    print(f"  Started: {parse_timestamp(ts)}")
    print(f"  Events: {len(events)}, User msgs: {len(user_msgs)}")
    print(f"  First prompt: {first_user}")
    print(f"  Source: {size:,} bytes -> Output: {out_path}")
    print(f"  Output size: {out_size:,} bytes")
    print()

print("All sessions exported.")
