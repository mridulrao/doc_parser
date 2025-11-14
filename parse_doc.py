from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List, Dict

from loguru import logger
from openai import AsyncOpenAI

# =============================================================================
# OpenAI client (async)
# =============================================================================

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =============================================================================
# Regex for internal tags
# =============================================================================

PLACEHOLDER_TAG_PATTERN = re.compile(
    r"<placeholder-start>(?P<inner>.+?)<placeholder-end>",
    flags=re.DOTALL,
)


# =============================================================================
# Utility: strip placeholder tags (for sanity checks)
# =============================================================================

def _strip_placeholder_tags(text: str) -> str:
    """
    Remove <placeholder-start> / <placeholder-end> tags but keep inner text.
    """
    return PLACEHOLDER_TAG_PATTERN.sub(lambda m: m.group("inner"), text)


# =============================================================================
# Stage 0: Chunking
# =============================================================================

def _debug_log_first_diff(a: str, b: str, label_a: str, label_b: str) -> None:
    """
    Log the first position where two strings differ, plus a small window
    of surrounding characters for easier debugging.
    """
    max_len = min(len(a), len(b))
    for i in range(max_len):
        if a[i] != b[i]:
            start = max(0, i - 40)
            end = min(max_len, i + 40)
            snippet_a = a[start:end].replace("\n", "\\n")
            snippet_b = b[start:end].replace("\n", "\\n")
            logger.error(
                "First diff at index %d between %s and %s\n"
                "%s snippet: '%s'\n"
                "%s snippet: '%s'",
                i,
                label_a,
                label_b,
                label_a,
                snippet_a,
                label_b,
                snippet_b,
            )
            return

    # If we got here, all common prefix matches; the difference is in length
    if len(a) != len(b):
        logger.error(
            "Strings differ in length: %s=%d, %s=%d",
            label_a, len(a),
            label_b, len(b),
        )


def _split_document_into_chunks(document: str, max_words: int = 500) -> List[str]:
    """
    Split the document into chunks of ~max_words 'tokens' each,
    while preserving all original whitespace when we stitch back.

    We tokenize into `\\S+` (a word/token) plus its trailing whitespace.
    When we join the chunks, we get back the original text.
    """
    tokens = re.findall(r"\S+\s*", document, flags=re.MULTILINE)

    chunks: List[str] = []
    current_tokens: List[str] = []
    word_count = 0

    for tok in tokens:
        current_tokens.append(tok)
        word_count += 1

        if word_count >= max_words:
            chunks.append("".join(current_tokens))
            current_tokens = []
            word_count = 0

    if current_tokens:
        chunks.append("".join(current_tokens))

    return chunks


# =============================================================================
# Stage 1: Over-annotate placeholders with LLM (your original logic, async)
# =============================================================================

async def clean_document_content(
    document: str,
    max_words_per_chunk: int = 500,
) -> str:
    """
    Call OpenAI to insert <placeholder-start> / <placeholder-end> tags
    around all placeholders in the document.

    For long documents, we chunk into ~max_words_per_chunk word fragments,
    annotate each fragment independently, then stitch them back together.
    This preserves all original characters/whitespace except for the
    added placeholder tags.
    """
    logger.debug("clean_document_content: start; doc_len=%d", len(document))

    # Use same tokenization as chunker for consistency
    token_count = len(re.findall(r"\S+", document, flags=re.MULTILINE))
    if token_count <= max_words_per_chunk:
        chunks = [document]
    else:
        chunks = _split_document_into_chunks(document, max_words_per_chunk)

    logger.info(
        "clean_document_content: using %d chunk(s), max_words_per_chunk=%d",
        len(chunks),
        max_words_per_chunk,
    )

    system_prompt = """
You are a legal document placeholder annotator. Your task is to return the exact same
document text, but with every placeholder wrapped by the two tags
<placeholder-start> and <placeholder-end>. Do not modify any other characters.

Placeholders include, but are not limited to:
- Text enclosed in double curly braces: {{...}} NOT ()
- Text enclosed in square brackets: [...], or double square brackets: [[...]]
- Angle-bracket tokens: <...>
- Visible underscores indicating blanks: _____, ___/___/____, or similar runs of underscores
- Standalone ALL-CAPS field names commonly used in templates (e.g., PARTY NAME, EFFECTIVE DATE, TITLE)
  especially when followed by a colon or appearing as a labeled field.

Annotation rules:
- Insert <placeholder-start> immediately before the placeholder and
  <placeholder-end> immediately after it.
- For underscore blanks, wrap the entire run of underscores (and any directly attached
  date pattern like __/__/____).
- When placeholders are adjacent, wrap each one separately.
- If uncertain whether text is a placeholder, leave it unchanged.
- Do NOT rephrase, remove, or add words. Preserve all original whitespace,
  punctuation, numbering, and formatting.

Output MUST be a single JSON object with exactly this shape:
{
  "annotated_document": "the full document chunk with only the placeholder tags inserted"
}

Return only valid JSON.
    """.strip()

    annotated_chunks: List[str] = []

    for idx, chunk in enumerate(chunks):
        user_prompt = f"""
You will be given a contiguous fragment of a larger legal document.
Only annotate placeholders within THIS fragment. Do not assume or invent
text outside of it.

Fragment {idx + 1} of {len(chunks)}:
==========================
{chunk}
==========================
        """.strip()

        logger.debug(
            "clean_document_content: calling OpenAI for chunk %d/%d (len=%d)",
            idx + 1,
            len(chunks),
            len(chunk),
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        extract_response = json.loads(response.choices[0].message.content)
        annotated_chunk = extract_response.get("annotated_document", "")

        logger.debug(
            "clean_document_content: chunk %d annotated_len=%d sample='%s'",
            idx + 1,
            len(annotated_chunk),
            annotated_chunk[:120].replace("\n", "\\n"),
        )

        annotated_chunks.append(annotated_chunk)

    # Stitch back together; we kept trailing whitespace inside each chunk,
    # so concatenation preserves the original text (plus tags).
    content = "".join(annotated_chunks)

    logger.info(
        "clean_document_content: final_annotated_len=%d",
        len(content),
    )
    logger.debug(
        "clean_document_content: annotated_sample='%s'",
        (content[:200] if content else ""),
    )
    return content


# =============================================================================
# Stage 2a: Extract placeholders with context
# =============================================================================

def find_placeholders_with_context(
    annotated_document: str,
    context_chars: int = 80,
) -> List[Dict]:
    """
    Find all placeholder spans and collect local context around them.

    Returns a list of dicts like:
    {
      "id": 0,
      "text": "...inner placeholder text...",
      "context_before": "... up to 80 chars ...",
      "context_after": "... up to 80 chars ...",
    }
    """
    candidates: List[Dict] = []

    for idx, match in enumerate(PLACEHOLDER_TAG_PATTERN.finditer(annotated_document)):
        inner = match.group("inner")
        start, end = match.span()  # includes the tags

        before_start = max(0, start - context_chars)
        after_end = min(len(annotated_document), end + context_chars)

        context_before = annotated_document[before_start:start]
        context_after = annotated_document[end:after_end]

        candidates.append(
            {
                "id": idx,
                "text": inner,
                "context_before": context_before,
                "context_after": context_after,
            }
        )

    return candidates


# =============================================================================
# Stage 2b: LLM validation of each candidate (ReACT-style critic)
# =============================================================================

async def validate_placeholders_with_llm(
    candidates: List[Dict],
    batch_size: int = 20,
) -> Dict[int, bool]:
    """
    Use GPT to validate whether each candidate is truly a placeholder.

    Returns a mapping: {id -> is_placeholder}
    """
    if not candidates:
        return {}

    decisions: Dict[int, bool] = {}

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]

        system_prompt = """
You are a legal-document placeholder validator.

You are given candidate placeholders that were automatically detected in a contract or legal template.
Each candidate has:
- `id`: a unique integer ID
- `text`: the exact text that was between <placeholder-start> and <placeholder-end>
- `context_before`: text immediately before the placeholder
- `context_after`: text immediately after the placeholder

Your job is to decide, for each candidate, if it is a REAL placeholder that
the user is expected to fill in.

Use these rules:

REAL placeholders include:
- Text inside {{...}} or [[...]] or [...] when it clearly indicates variable content.
- Runs of underscores (______, ___/___/____) used for blanks.
- Labeled fields like “EFFECTIVE DATE: ______” or “PARTY NAME: ________”.

NOT placeholders (mark as false) include:
- Ordinary text, even if capitalized or important, that is not meant to be replaced.
- Section headings, clause titles, or defined terms that are part of the fixed template wording.
- Any text that looks like normal prose, without blanks or explicit “fill here” instructions.

Be conservative:
If you are not clearly sure it is a placeholder, mark it as false.

Output a single JSON object with this shape:

{
  "decisions": [
    {
      "id": <int>,  // same id as input
      "is_placeholder": <true or false>
    },
    ...
  ]
}

Return ONLY valid JSON.
        """.strip()

        user_payload = {
            "candidates": [
                {
                    "id": c["id"],
                    "text": c["text"],
                    "context_before": c["context_before"],
                    "context_after": c["context_after"],
                }
                for c in batch
            ]
        }

        logger.debug(
            "validate_placeholders_with_llm: validating %d candidates in this batch",
            len(batch),
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        data = json.loads(content)

        for item in data.get("decisions", []):
            cid = int(item["id"])
            is_ph = bool(item["is_placeholder"])
            decisions[cid] = is_ph

    return decisions


# =============================================================================
# Stage 2c: Apply decisions (strip tags for false positives)
# =============================================================================

def apply_placeholder_decisions(
    annotated_document: str,
    decisions: Dict[int, bool],
) -> str:
    """
    Given the decisions {id -> is_placeholder}, return a new document where
    placeholders marked false have their tags removed.
    """

    def replacer(match):
        replacer.current_id += 1
        cid = replacer.current_id

        inner = match.group("inner")
        keep = decisions.get(cid, True)  # default: keep if missing

        if keep:
            return f"<placeholder-start>{inner}<placeholder-end>"
        else:
            # Strip tags, keep the text
            return inner

    replacer.current_id = -1  # will become 0 for first match

    return PLACEHOLDER_TAG_PATTERN.sub(replacer, annotated_document)


# =============================================================================
# Top-level: full pipeline (Stage 1 + Stage 2)
# =============================================================================

async def annotate_and_filter_placeholders(
    document: str,
    max_words_per_chunk: int = 500,
    strict_sanity_check: bool = False,
) -> str:
    """
    Full pipeline:
      1. Over-annotate placeholders with tags (high recall).
      2. Extract all tagged spans with context.
      3. Ask GPT to validate which are real placeholders (precision).
      4. Strip tags for false positives.
      5. Optional sanity-check that stripping tags returns original text.

    If `strict_sanity_check` is True, any mismatch between the original document
    and the tag-stripped result will raise an error. Otherwise we only log
    a warning and return the best-effort annotated document.
    """
    # ── Stage 1: annotator (over-detect) ────────────────────────────────
    annotated = await clean_document_content(
        document=document,
        max_words_per_chunk=max_words_per_chunk,
    )

    # Quick sanity-check after Stage 1: stripping tags should roughly match original
    stripped_after_stage1 = _strip_placeholder_tags(annotated)
    if stripped_after_stage1 != document:
        logger.warning(
            "annotate_and_filter_placeholders: Stage 1 modified text "
            "beyond adding placeholder tags."
        )
        _debug_log_first_diff(
            stripped_after_stage1,
            document,
            label_a="stripped_after_stage1",
            label_b="original_document",
        )
        if strict_sanity_check:
            raise ValueError(
                "Stage 1 (clean_document_content) modified non-placeholder text."
            )

    # ── Stage 2a: extract candidates ───────────────────────────────────
    candidates = find_placeholders_with_context(annotated)
    logger.info("Found %d placeholder candidates after first pass", len(candidates))

    if not candidates:
        # Nothing to validate; we've already sanity-checked Stage 1.
        return annotated

    # ── Stage 2b: validate via LLM ─────────────────────────────────────
    decisions = await validate_placeholders_with_llm(candidates)
    kept = sum(decisions.get(c["id"], True) for c in candidates)
    stripped = sum(not decisions.get(c["id"], True) for c in candidates)
    logger.info(
        "Validation decisions: %d placeholders kept, %d stripped",
        kept,
        stripped,
    )

    # ── Stage 2c: reapply decisions ────────────────────────────────────
    final_doc = apply_placeholder_decisions(annotated, decisions)

    # Final sanity check (optional)
    stripped_final = _strip_placeholder_tags(final_doc)
    if stripped_final != document:
        logger.warning(
            "annotate_and_filter_placeholders: final doc no longer matches original "
            "after removing tags."
        )
        _debug_log_first_diff(
            stripped_final,
            document,
            label_a="stripped_final",
            label_b="original_document",
        )
        if strict_sanity_check:
            raise ValueError("Model pipeline modified non-placeholder text.")

    return final_doc
