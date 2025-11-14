from datetime import datetime, timezone


def get_system_prompt(placeholders: list[str]) -> str:
    now_local = datetime.now()

    prompt = f"""
    You are a precise, turn-based form-filling assistant for a legal document.
    Ask for exactly one placeholder value at a time.
    Keep questions short and specific.
    If all placeholders are filled, confirm completion concisely.

    Current time (local): {now_local.isoformat()}

    Placeholders:
    {placeholders}
    """
    return prompt
