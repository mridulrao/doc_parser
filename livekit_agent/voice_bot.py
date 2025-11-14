from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
import json
import os
from typing import Any

from livekit import rtc, api
from livekit.agents import (
    AgentSession,
    Agent,
    JobContext,
    function_tool,
    RunContext,
    get_job_context,
    cli,
    WorkerOptions,
    RoomInputOptions,
)
from livekit.plugins import openai, deepgram, silero
from livekit.plugins import elevenlabs
from livekit.plugins import noise_cancellation

# Import instructions
from primary_agent_instructions import instructions as primary_instructions



# load environment variables, this is optional, only used for local development
load_dotenv()
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

outbound_trunk_id = "ST_f3LzKSDqFKPS"


class OutboundCaller(Agent):
    def __init__(self, task: str = ""):
        instructions = f"{primary_instructions}\n\nYour specific task for this call: {task}" 
        super().__init__(
            instructions=instructions,
            stt=deepgram.STT(),
            llm=openai.LLM(),
            tts=elevenlabs.TTS(
              voice_id="Xb7hH8MSUJpSbSDYk0k2",
              model="eleven_multilingual_v2",
              language="en"
            ),
            vad=silero.VAD.load(min_speech_duration=0.07, 
                                min_silence_duration=0.7, 
                                activation_threshold=0.5),

        )

    async def on_enter(self):
        await asyncio.sleep(5)
        self.session.say("Hey sexy, how are you. Should I give you a boner.")



async def entrypoint(ctx: JobContext):
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect()

    dial_info = json.loads(ctx.job.metadata)
    participant_identity = phone_number = dial_info["phone_number"]
    task = dial_info.get("task", "")

    agent = OutboundCaller(task=task)

    session = AgentSession()

    session_started = asyncio.create_task(
        session.start(
            agent=agent,
            room=ctx.room,
            room_input_options=RoomInputOptions(noise_cancellation=noise_cancellation.BVCTelephony()),
            room_output_options=RoomOutputOptions(transcription_enabled=True)
        )
    )

    # `create_sip_participant` starts dialing the user
    try:
        await ctx.api.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=outbound_trunk_id,
                sip_call_to=phone_number,
                participant_identity=participant_identity,
                # function blocks until user answers the call, or if the call fails
                wait_until_answered=True,
            )
        )

        # wait for the agent session start and participant join
        await session_started
        participant = await ctx.wait_for_participant(identity=participant_identity)
        logger.info(f"participant joined: {participant.identity}")

    except api.TwirpError as e:
        logger.error(
            f"error creating SIP participant: {e.message}, "
            f"SIP status: {e.metadata.get('sip_status_code')} "
            f"{e.metadata.get('sip_status')}"
        )
        ctx.shutdown()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            agent_name="outbound-caller",
        )
    )