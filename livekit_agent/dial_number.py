import json
from datetime import datetime
from livekit import api
from dotenv import load_dotenv
from .voice_bot_instructions import get_system_prompt

load_dotenv()

def build_system_prompt_for_call(
    placeholders: list[str],
) -> str:
    return get_system_prompt(placeholders)

async def make_call(phone_number: str, task: str):
    lkapi = api.LiveKitAPI()
    room_name = f"outbound-call-{phone_number.replace('+', '')}"
    metadata = json.dumps({
        "phone_number": phone_number,
        "task": task
    })
    dispatch = await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(
            agent_name="outbound-caller",
            room=room_name,
            metadata=metadata
        )
    )
    await lkapi.aclose()
    return dispatch

async def generate_prompt_and_call(
    phone_number: str,
    placeholders: list[str],
):
    prompt = build_system_prompt_for_call(placeholders)
    print(f"Prompt: {prompt}")
    return await make_call(phone_number, prompt)