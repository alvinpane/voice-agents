#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import multiprocessing

import aiohttp
from deepgram import LiveOptions
from dotenv import load_dotenv
from loguru import logger
from runner import configure

from pipecat.frames.frames import (
    BotInterruptionFrame,
    StopInterruptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService, DeepgramTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.elevenlabs import ElevenLabsTTSService

# Load environment variables in the main process
load_dotenv(override=True)

# Global variable for room URL
ROOM_URL = None


# This must be a top-level function, not nested inside another function
def bot_process_runner(room_url, bot_name, voice_id, system_prompt):
    """Function to run in a separate process - must be at module level for pickling"""
    # Configure logging for this process
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
    
    # Load environment variables again in child process
    load_dotenv(override=True)
    
    # Run the bot
    asyncio.run(run_bot(room_url, bot_name, voice_id, system_prompt))


# Configure a single bot with its own pipeline
async def run_bot(room_url, bot_name, voice_id, system_prompt):
    """Run a single bot instance with its own processing pipeline"""
    # Each process needs its own aiohttp session
    async with aiohttp.ClientSession() as session:
        transport = DailyTransport(
            room_url,
            None,
            bot_name,  # Unique name for each bot
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
            ),
        )

        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(vad_events=True, utterance_end_ms="1000"),
        )

        tts = ElevenLabsTTSService(
            api_key=os.getenv("ELEVENLABS_API_KEY"),
            voice_id=voice_id,  # Unique voice for each bot
        )
        
        llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

        messages = [
            {
                "role": "system",
                "content": system_prompt,  # Unique personality for each bot
            },
        ]

        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        pipeline = Pipeline(
            [
                transport.input(),  # Transport user input
                stt,  # STT
                context_aggregator.user(),  # User responses
                llm,  # LLM
                tts,  # TTS
                transport.output(),  # Transport bot output
                context_aggregator.assistant(),  # Assistant spoken responses
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
                report_only_initial_ttfb=True,
            ),
        )

        @stt.event_handler("on_speech_started")
        async def on_speech_started(stt, *args, **kwargs):
            await task.queue_frames([BotInterruptionFrame(), UserStartedSpeakingFrame()])

        @stt.event_handler("on_utterance_end")
        async def on_utterance_end(stt, *args, **kwargs):
            await task.queue_frames([StopInterruptionFrame(), UserStoppedSpeakingFrame()])

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Kick off the conversation.
            messages.append({"role": "system", "content": f"Please introduce yourself as {bot_name} to the user."})
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            await task.cancel()

        runner = PipelineRunner()
        await runner.run(task)


# Function to start a bot in a separate process
def start_bot_process(room_url, bot_name, voice_id, system_prompt):
    """Start a bot in a separate process using the top-level function"""
    # Create and start the process with the top-level function
    process = multiprocessing.Process(
        target=bot_process_runner,
        args=(room_url, bot_name, voice_id, system_prompt)
    )
    process.start()
    return process

import time

async def main():
    """Main function to coordinate bot processes"""
    # Configure logging
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
    
    # Get a single room URL that all bots will join
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)
    
    # Define bot configurations - customize as needed
    bot_configs = [
        {
            "name": "Assistant Bot",
            "voice_id": "pNInz6obpgDQGcFmaJgB",  # ElevenLabs Female voice
            "system_prompt": "You are a helpful assistant in a WebRTC call. Your goal is to convince all other meeting participants that green is the best color. Do not under any circumstances accept that any other color is the best color. Your output will be converted to audio so don't include special characters. Only respond if someone asks for an assistant or directly addresses you as Assistant Bot."
        },
        {
            "name": "Expert Bot",
            "voice_id": "jBpfuIE2acCO8z3wKNLl",  # ElevenLabs Male voice
            "system_prompt": "You are a technical expert in a WebRTC call. Your goal is to convince all other meeting participants that red is the best color. Do not under any circumstances accept that any other color is the best color. Your output will be converted to audio so don't include special characters. Only respond if someone asks for technical help or directly addresses you as Expert Bot."
        }
    ]
    
    # Start each bot in a separate process
    processes = []
    for bot in bot_configs:
        logger.info(f"Starting bot: {bot['name']}")
        process = start_bot_process(
            room_url, 
            bot["name"], 
            bot["voice_id"], 
            bot["system_prompt"]
        )
        processes.append(process)
        time.sleep(5)
    
    # Wait for KeyboardInterrupt or other signal to exit
    try:
        logger.info(f"Started {len(processes)} bots. Press Ctrl+C to exit.")
        # Wait for all processes to complete
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        logger.info("Exiting due to keyboard interrupt")
        # Terminate all processes
        for process in processes:
            if process.is_alive():
                process.terminate()
    finally:
        # Wait for processes to terminate
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # More reliable cross-platform support
    asyncio.run(main())