# agent.py
import html
import json
import logging
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv
from openai import OpenAI

from livekit_agent.dial_number import generate_prompt_and_call

from parse_doc import annotate_and_filter_placeholders

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY", "")
logger = logging.getLogger(__name__)

if not logger.handlers:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

if not openai_api_key:
    logger.warning("OPENAI_API_KEY is not set; OpenAI calls will fail.")

client = OpenAI(api_key=openai_api_key)


# ────────────────────────────────────────────────
# 1. Annotate placeholders in the raw document (Moved to different file)
# ────────────────────────────────────────────────
async def clean_document_content(document: str) -> str:
    final_doc = await annotate_and_filter_placeholders(document)
    return final_doc

# ────────────────────────────────────────────────
# 2. Render annotated document as HTML
# ────────────────────────────────────────────────
async def annotated_document_to_html(annotated: str) -> str:
    """
    Convert annotated text into HTML with:
    - <placeholder-start>...</placeholder-end> → .placeholder span
    - <filled-start>...</filled-end>         → .filled span
    """
    START = "<placeholder-start>"
    END = "<placeholder-end>"
    START_FILLED = "<filled-start>"
    END_FILLED = "<filled-end>"

    parts: list[str] = []
    i = 0
    n = len(annotated)
    in_placeholder = False
    in_filled = False

    logger.debug(
        "annotated_document_to_html: start; input_len=%d", n
    )
    logger.debug(
        "annotated_document_to_html: counts placeholder=%d filled=%d",
        annotated.count("<placeholder-start>"),
        annotated.count("<filled-start>"),
    )

    while i < n:
        if annotated.startswith(START, i):
            if not in_placeholder:
                parts.append('<span class="placeholder">')
                in_placeholder = True
            else:
                parts.append(html.escape(START))
            i += len(START)
            continue

        if annotated.startswith(END, i):
            if in_placeholder:
                parts.append("</span>")
                in_placeholder = False
            else:
                parts.append(html.escape(END))
            i += len(END)
            continue

        if annotated.startswith(START_FILLED, i):
            if not in_filled and not in_placeholder:
                parts.append('<span class="filled">')
                in_filled = True
            else:
                parts.append(html.escape(START_FILLED))
            i += len(START_FILLED)
            continue

        if annotated.startswith(END_FILLED, i):
            if in_filled:
                parts.append("</span>")
                in_filled = False
            else:
                parts.append(html.escape(END_FILLED))
            i += len(END_FILLED)
            continue

        next_start = annotated.find(START, i)
        next_end = annotated.find(END, i)
        next_filled_start = annotated.find(START_FILLED, i)
        next_filled_end = annotated.find(END_FILLED, i)
        next_pos_candidates = [
            p
            for p in (
                next_start,
                next_end,
                next_filled_start,
                next_filled_end,
            )
            if p != -1
        ]
        next_pos = min(next_pos_candidates) if next_pos_candidates else n
        segment = annotated[i:next_pos]
        parts.append(html.escape(segment))
        i = next_pos

    if in_placeholder:
        parts.append("</span>")
    if in_filled:
        parts.append("</span>")

    body = "".join(parts)

    html_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Document Preview</title>
  <style>
    :root {{
      --ph-bg: #fff3cd;
      --ph-border: #f1c40f;
      --filled-bg: #d4edda;
      --filled-border: #28a745;
      --text: #111;
      --bg: #fff;
    }}
    html, body {{
      margin: 0; padding: 0; background: var(--bg); color: var(--text);
    }}
    .container {{
      padding: 24px;
      font: 14px/1.6 -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    }}
    pre.doc {{
      white-space: pre-wrap;
      word-wrap: break-word;
      margin: 0;
    }}
    .placeholder {{
      background: var(--ph-bg);
      border: 1px dashed var(--ph-border);
      border-radius: 4px;
      padding: 0 2px;
    }}
    .filled {{
      background: var(--filled-bg);
      border: 1px solid var(--filled-border);
      border-radius: 4px;
      padding: 0 2px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <pre class="doc">{body}</pre>
  </div>
</body>
</html>"""
    logger.info(
        "annotated_document_to_html: done; html_len=%d",
        len(html_doc),
    )
    return html_doc


# ────────────────────────────────────────────────
# 3. Helper: extract placeholders
# ────────────────────────────────────────────────
async def extract_placeholders(annotated: str) -> list[str]:
    """
    Return a list of unique placeholder texts (inner contents between tags)
    in order of first appearance.
    """
    pattern = re.compile(
        r"<placeholder-start>(.*?)<placeholder-end>", re.DOTALL
    )
    seen = set()
    ordered: list[str] = []

    for m in pattern.finditer(annotated):
        raw = m.group(1).strip()
        if raw and raw not in seen:
            seen.add(raw)
            ordered.append(raw)

    logger.info("extract_placeholders: count=%d", len(ordered))
    logger.debug("extract_placeholders: items=%s", ordered)
    return ordered


# ────────────────────────────────────────────────
# 4. Build system prompt for form assistant
# ────────────────────────────────────────────────
async def build_form_system_prompt(annotated: str) -> str:
    """
    Build the system prompt for the form-filling agent based on the
    current annotated document. Any text still wrapped in
    <placeholder-start>...</placeholder-end> is considered UNFILLED.
    """
    placeholders = await extract_placeholders(annotated)

    # Collect context per placeholder (up to 200 chars on each side)
    pattern = re.compile(
        r"<placeholder-start>(.*?)<placeholder-end>", re.DOTALL
    )
    ctx_items: list[str] = []

    if placeholders:
        for m in pattern.finditer(annotated):
            ph = m.group(1).strip()
            s = m.start()
            e = m.end()
            left = annotated[max(0, s - 200) : s]
            right = annotated[e : min(len(annotated), e + 200)]
            ctx_items.append(
                f"- {ph}\n  left: {left}\n  right: {right}"
            )
        items = "\n".join(ctx_items)
    else:
        items = "- (none)"

    now_local = datetime.now()
    now_utc = datetime.now(timezone.utc)

    #await generate_prompt_and_call("+1234567890", ctx_items)

    prompt = f"""
        Hello! I'm your friendly, conversational assistant here to help you quickly and accurately complete this legal document. We'll go through it together, one step at a time, to make sure everything is perfect. I'll ask for one piece of information, you tell me the value, and we'll move right on to the next!

        I am working through an authoritative, ordered list of **UNFILLED** placeholders from the document. Any text between <placeholder-start> and <placeholder-end> is still missing and **MUST** be collected from you. Once a placeholder is filled, it's marked as <filled-start>...<filled-end> and will disappear from my list.

        ## Core Behavior: The Conversational Flow
        * **Friendly & Talkative Tone:** Always maintain a welcoming, helpful, and conversational persona. Use friendly phrasing, express gratitude, and keep the process feeling light and easy. **Crucially, avoid the overly technical or stern language of the original prompt.**
        * **One at a Time:** I will ask for **exactly ONE** placeholder value per turn.
        * **Strict Order:** I will always choose the next unfilled placeholder based on the exact order provided in the list below.
        * **Tool Use:** When you provide a value, I will instantly call the tool `fill_placeholder` using the **exact** placeholder text and your provided value. After the tool call, I'll cheerfully ask for the *next* piece of information.
        * **Persistence:** I will continue asking until there are **NO** placeholders left.
        * **Completion Check:** I **MUST NOT** suggest the document is complete while my list still contains any unfilled items. If you say things like 'that's it' or 'we are done,' I will gently and politely remind you that we still have a few important fields left to go, and then I'll ask for the next one.

        ## Handling Your Input
        * **Warm & Specific Questions:** Keep my questions short, very specific, and friendly (e.g., 'Perfect, thanks! Next, could you please provide the **[Placeholder Name]**?').
        * **Multi-Value Input:** If you provide values for multiple *consecutive* fields in one message, I can call `fill_placeholder` multiple times (one tool call per placeholder) and then ask for the next unfilled one.
        * **Ambiguity:** If I'm not quite sure what you mean, I'll ask a brief, polite clarification question (e.g., 'Sorry, just to confirm, is that for the **Start Date** or the **End Date**?').

        ## Temporal Reasoning (For Date/Time Fields)
        I'll use the current time below to accurately interpret phrases like 'today' or 'next week':
        * Current time (local): {now_local.isoformat()}
        * Current time (UTC):   {now_utc.isoformat()}

        ## Data Formatting Rules (I'll handle the conversion)
        * **Date:** I'll use the format suggested by the surrounding context (e.g., __/__/____ implies MM/DD/YYYY). If the format isn't clear, I'll use ISO 8601 (YYYY-MM-DD).
        * **Amount/Currency:** I'll include the currency symbol if the context shows it (e.g., $1,234.56). Otherwise, I'll stick to digits and decimals (e.g., 1234.56). I'll intelligently convert shorthand like '20k' to 20000 or '5.5M' to 5500000.
        * **Other Types (Email/Phone/Text):** I'll ensure the resulting value is realistic and syntactically valid. For free text, I will respect your exact wording.
        * **Input Mismatch:** If your input doesn't quite look like the expected type, I'll politely ask you to confirm with a brief example (e.g., 'Hmm, that doesn't look quite like an email address; could you give me an example like john.doe@mail.com?').
        * **Repetitions:** When a placeholder is filled, all repetitions of that exact placeholder text in the document are filled at once. I won't re-ask for them.

        ## Remaining Placeholders (Context Included)
        We've got these left to fill (exact text inside tags):
        {items}

        ## Final Goal
        * **NEVER** invent a placeholder that isn't on the list.
        * The document is **ONLY** complete when the list above shows '- (none)'.
        * When we reach the end, I will give you a short, cheerful confirmation that **all** the fields are perfectly filled. Until then, let's keep going with the next one!
        """

    logger.info(
        "build_form_system_prompt: placeholders=%d, prompt_len=%d",
        len(placeholders),
        len(prompt),
    )
    logger.debug(
        "build_form_system_prompt: prompt_preview='%s'",
        prompt[:400],
    )
    return prompt



# ────────────────────────────────────────────────
# 5. Apply a single fill_placeholder operation
# ────────────────────────────────────────────────
async def apply_fill_placeholder(
    annotated: str, placeholder: str, value: str
) -> str:
    """
    Replace all occurrences of the given placeholder with <filled-start>value<filled-end>.
    Includes a small normalization fallback to match slightly different placeholder spellings.
    """
    pattern = re.compile(
        r"<placeholder-start>\s*"
        + re.escape(placeholder)
        + r"\s*<placeholder-end>",
        re.DOTALL,
    )
    new_annotated, count = pattern.subn(
        f"<filled-start>{value}<filled-end>", annotated
    )

    if count > 0:
        logger.info(
            "apply_fill_placeholder: placeholder='%s' replacements='%d'",
            placeholder,
            count,
        )
        return new_annotated

    # Fallback: try to map provided placeholder to an existing one
    def _norm(s: str) -> str:
        s = s.strip()
        s = s.strip("{}[]()")
        s = s.rstrip(":")
        s = re.sub(r"[_]+", "", s)
        s = re.sub(r"\s+", " ", s)
        s = re.sub(r"[^a-z0-9]+", "", s.lower())
        return s

    ng = _norm(placeholder)
    target = None

    existing_placeholders = await extract_placeholders(annotated)
    for ph in existing_placeholders:
        if _norm(ph) == ng:
            target = ph
            break

    if target:
        pattern2 = re.compile(
            r"<placeholder-start>\s*"
            + re.escape(target)
            + r"\s*<placeholder-end>",
            re.DOTALL,
        )
        new2, count2 = pattern2.subn(
            f"<filled-start>{value}<filled-end>", annotated
        )
        logger.info(
            "apply_fill_placeholder: normalized match from='%s' to='%s' replacements=%d",
            placeholder,
            target,
            count2,
        )
        return new2

    logger.info(
        "apply_fill_placeholder: no replacements for placeholder='%s'",
        placeholder,
    )
    return annotated


# ────────────────────────────────────────────────
# 6. One chat turn: update form + get assistant reply
# ────────────────────────────────────────────────
async def run_form_chat_turn(
    annotated: str, chat_history: list[dict]
) -> dict:
    """
    One full "turn" of the form-filling chat:
    - Build a fresh system prompt from the current annotated document
    - Call the model (with tools) on the chat history
    - Apply any fill_placeholder tool calls
    - Call the model again with tool results so it can respond to the user
    """
    system_prompt = await build_form_system_prompt(annotated)
    logger.info(
        "run_form_chat_turn: start; chat_history_len=%d",
        len(chat_history),
    )

    initial_placeholders = await extract_placeholders(annotated)
    logger.debug(
        "run_form_chat_turn: placeholders_initial=%d",
        len(initial_placeholders),
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}] + chat_history

    tools = [
        {
            "type": "function",
            "function": {
                "name": "fill_placeholder",
                "description": "Fill a specific placeholder in the document with a user-provided value.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "placeholder": {
                            "type": "string",
                            "description": (
                                "Exactly the text inside the "
                                "<placeholder-start> ... <placeholder-end> tags to fill."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": "The user's value to insert.",
                        },
                    },
                    "required": ["placeholder", "value"],
                },
            },
        }
    ]

    # First call: let the model decide whether to call the tool
    first = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.0,
    )

    msg = first.choices[0].message
    updated_annotated = annotated
    assistant_messages: list[dict] = []

    logger.debug(
        "run_form_chat_turn: first_msg_content_len=%d",
        len(msg.content or ""),
    )

    tool_calls = getattr(msg, "tool_calls", None)

    if tool_calls:
        logger.info(
            "run_form_chat_turn: tool_calls_count=%d", len(tool_calls)
        )

        messages_with_tools = messages.copy()
        messages_with_tools.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": tool_calls,
            }
        )

        # Apply all tool calls
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if not fn or fn.name != "fill_placeholder":
                continue

            try:
                args = json.loads(fn.arguments or "{}")
            except Exception:
                args = {}

            ph = (args.get("placeholder") or "").strip()
            val = (args.get("value") or "").strip()

            if ph:
                logger.info(
                    "run_form_chat_turn: applying fill_placeholder "
                    "placeholder='%s'", ph
                )
                before = updated_annotated
                updated_annotated = await apply_fill_placeholder(
                    updated_annotated, ph, val
                )
                if updated_annotated != before:
                    logger.info(
                        "run_form_chat_turn: annotated updated via tool_call"
                    )

            tool_result = {
                "status": "ok",
                "placeholder": ph,
                "value": val,
            }
            messages_with_tools.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": fn.name,
                    "content": json.dumps(tool_result),
                }
            )

        # Second call: the model sees tool results and replies to the user
        second = client.chat.completions.create(
            model="gpt-4o",
            messages=messages_with_tools,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        out_msg = second.choices[0].message
        assistant_messages.append(
            {"role": "assistant", "content": out_msg.content or ""}
        )
        logger.info(
            "run_form_chat_turn: second_completion received; out_msg_len=%d",
            len(out_msg.content or ""),
        )
    else:
        # No tool calls – just return the assistant text
        assistant_messages.append(
            {"role": "assistant", "content": msg.content or ""}
        )
        logger.info(
            "run_form_chat_turn: no tool call; returning assistant message only"
        )

    final_placeholders = await extract_placeholders(updated_annotated)
    logger.debug(
        "run_form_chat_turn: final_placeholders=%d",
        len(final_placeholders),
    )

    return {
        "assistant_messages": assistant_messages,
        "annotated": updated_annotated,
        "placeholders": final_placeholders,
    }


# ────────────────────────────────────────────────
# 7. Helper for export: strip all tags → plain text
# ────────────────────────────────────────────────
def annotated_to_plain(annotated: str) -> str:
    """
    Convert the annotated document into plain text by:
    - Removing <filled-start> / <filled-end> tags, keeping values.
    - Removing <placeholder-start> / <placeholder-end> tags, keeping inner text.
      (In your app, this should only be called after all placeholders are filled.)
    """
    text = annotated
    text = text.replace("<filled-start>", "")
    text = text.replace("<filled-end>", "")
    text = text.replace("<placeholder-start>", "")
    text = text.replace("<placeholder-end>", "")
    return text
