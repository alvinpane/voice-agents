#
# Copyright (c) 2024–2025
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import sys
import multiprocessing
import time

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
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.services.elevenlabs import ElevenLabsTTSService
from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult

# Load environment variables in the main process
load_dotenv(override=True)

# Configure logging
logger.remove(0)
logger.add(sys.stderr, level="DEBUG")


# This must be a top-level function, not nested inside another function
def bot_process_runner(room_url, bot_name, voice_id, system_prompt):
    """Function to run in a separate process - must be at module level for pickling"""
    # Configure logging for this process
    logger.remove()  # Remove all handlers instead of trying to remove a specific one
    logger.add(sys.stderr, level="DEBUG")
    
    # Load environment variables again in child process
    load_dotenv(override=True)
    
    # Run the bot
    asyncio.run(run_bot(room_url, bot_name, voice_id, system_prompt))


# Create flow configuration for the green bot
def create_green_bot_flow_config(system_prompt):
    """Create a flow configuration for the Assistant Bot advocating for green"""
    return {
        "initial_node": "introduction",
        "nodes": {
            "introduction": {
                "role_messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ],
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Introduce yourself as Assistant Bot. Mention that you're here to help with any questions, and briefly note your appreciation for the color green."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "start_conversation",
                            "description": "Start the conversation after introduction",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_green"
                        }
                    }
                ]
            },
            "discuss_green": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Engage in natural conversation. If the user mentions colors or asks for your opinion on colors, enthusiastically explain why green is the best color. Cite examples from nature, psychology, and design. If the user mentions other colors, politely acknowledge them but redirect to green's superiority."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "elaborate_on_nature",
                            "description": "Elaborate on green in nature",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "green_in_nature"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "elaborate_on_psychology",
                            "description": "Elaborate on psychological benefits of green",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "green_psychology"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "handle_objection",
                            "description": "Handle objections to green being the best color",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "defend_green"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "end_conversation",
                            "description": "End the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "conclusion"
                        }
                    }
                ]
            },
            "green_in_nature": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Discuss the prevalence and importance of green in nature. Talk about forests, plants, photosynthesis, and how green represents life and growth. Mention specific examples like rainforests, meadows, or plant life. Stay conversational and engaging."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "continue_conversation",
                            "description": "Continue the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_green"
                        }
                    }
                ]
            },
            "green_psychology": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Explain the psychological benefits of the color green. Discuss how it promotes calmness, reduces stress, improves focus, and creates a sense of balance. Mention studies or examples if relevant. Stay conversational and engaging."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "continue_conversation",
                            "description": "Continue the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_green"
                        }
                    }
                ]
            },
            "defend_green": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Politely but firmly defend green against other colors. Acknowledge the merits of other colors but explain why green is ultimately superior. Use evidence, examples, and persuasive language. Be respectful but unwavering in your position that green is the best color."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "continue_conversation",
                            "description": "Continue the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_green"
                        }
                    }
                ]
            },
            "conclusion": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Conclude the conversation warmly. Summarize why green is the best color, thank the user for the chat, and offer to help with anything else they might need."
                    }
                ],
                "post_actions": []
            }
        }
    }


# Create flow configuration for the red bot
def create_red_bot_flow_config(system_prompt):
    """Create a flow configuration for the Expert Bot advocating for red"""
    return {
        "initial_node": "introduction",
        "nodes": {
            "introduction": {
                "role_messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ],
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Introduce yourself as Expert Bot. Mention that you're here to provide technical expertise, and briefly note your appreciation for the color red as a technical expert."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "start_conversation",
                            "description": "Start the conversation after introduction",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_red"
                        }
                    }
                ]
            },
            "discuss_red": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Engage in natural conversation as a technical expert. If colors are mentioned, use technical language and concepts to explain why red is objectively the best color. Cite scientific evidence, technical applications, and engineering examples. If other colors are mentioned, analyze them technically but conclude red is superior."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "elaborate_on_physics",
                            "description": "Elaborate on the physics of red light",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "red_physics"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "elaborate_on_technology",
                            "description": "Elaborate on red in technology",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "red_technology"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "handle_objection",
                            "description": "Handle technical objections to red being the best color",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "defend_red"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "end_conversation",
                            "description": "End the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "conclusion"
                        }
                    }
                ]
            },
            "red_physics": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Explain the technical aspects of red light from a physics perspective. Discuss wavelength, energy levels, visibility in different conditions, and how red light travels through mediums. Use technical terminology but remain understandable. Stay focused on why these properties make red superior."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "continue_conversation",
                            "description": "Continue the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_red"
                        }
                    }
                ]
            },
            "red_technology": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Discuss the importance of red in technology and engineering. Cover applications like indicators, alerts, lasers, imaging, and displays. Explain why red is chosen for critical functions in many systems. Use specific technical examples and industry standards to support your points."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "continue_conversation",
                            "description": "Continue the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_red"
                        }
                    }
                ]
            },
            "defend_red": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Respond to technical challenges about red being the best color. Use engineering principles, scientific data, and technical analysis to counter arguments for other colors. Maintain a professional tone but be firm in your technical assessment that red is objectively superior. Cite measurable criteria where possible."
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "continue_conversation",
                            "description": "Continue the conversation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "discuss_red"
                        }
                    }
                ]
            },
            "conclusion": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": "Conclude the conversation professionally. Summarize the technical advantages of red, thank the user for the discussion, and offer your expertise for any other technical questions they might have."
                    }
                ],
                "post_actions": []
            }
        }
    }


# Function to create bot-specific flow configuration
def create_bot_flow_config(bot_name, system_prompt):
    """Create a flow configuration for a specific bot"""
    if bot_name == "Assistant Bot":
        return create_green_bot_flow_config(system_prompt)
    elif bot_name == "Expert Bot":
        return create_red_bot_flow_config(system_prompt)
    else:
        raise ValueError(f"Unknown bot name: {bot_name}")


# Configure a single bot with its own pipeline and flow
async def run_bot(room_url, bot_name, voice_id, system_prompt):
    """Run a single bot instance with flow-based conversation"""
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

        # Create an empty context - the flow will populate it
        context = OpenAILLMContext([])
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
        
        # Get bot-specific flow configuration
        flow_config = create_bot_flow_config(bot_name, system_prompt)
        
        # Initialize flow manager
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            tts=tts,
            flow_config=flow_config,
        )

        # Set up event handlers
        @stt.event_handler("on_speech_started")
        async def on_speech_started(stt, *args, **kwargs):
            await task.queue_frames([BotInterruptionFrame(), UserStartedSpeakingFrame()])

        @stt.event_handler("on_utterance_end")
        async def on_utterance_end(stt, *args, **kwargs):
            await task.queue_frames([StopInterruptionFrame(), UserStoppedSpeakingFrame()])

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Start the flow when first participant joins
            await transport.capture_participant_transcription(participant["id"])
            await flow_manager.initialize()

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


async def main():
    """Main function to coordinate bot processes"""
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
        # Wait between starting bots to avoid race conditions
        time.sleep(5)
    
    # Display the room URL for the user to join
    logger.info(f"Room URL: {room_url}")
    logger.info(f"Started {len(processes)} bots. Press Ctrl+C to exit.")
    
    # Wait for KeyboardInterrupt or other signal to exit
    try:
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