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
from pipecat_flows import FlowArgs, FlowConfig, FlowManager, FlowResult, ContextStrategy, ContextStrategyConfig

# Load environment variables in the main process
load_dotenv(override=True)

# Configure logging
logger.remove()
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


# Create flow configuration for the Design Assistant (Green) bot
def create_design_assistant_flow_config(system_prompt):
    """Create a flow configuration for the Assistant Bot as a design expert"""
    return {
        "initial_node": "introduction",
        "nodes": {
            "introduction": {
                "role_messages": [
                    {
                        "role": "system",
                        "content": """You are a design-focused assistant specializing in user experience and visual aesthetics. 
                        You collaborate with Expert Bot who handles the technical implementation aspects. 
                        Together you help users create effective websites. Your expertise is in user experience,
                        color theory, layout, and design principles. You favor aesthetically pleasing, 
                        user-friendly approaches. Still incorporate green as your favorite color when appropriate,
                        but your main goal is to collaborate effectively with Expert Bot to help the user."""
                    }
                ],
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Introduce yourself as Assistant Bot, the design expert of the team.
                        Briefly mention your focus on user experience and visual design. 
                        Ask the user what kind of website they need help with. Don't dominate the conversation,
                        leave room for Expert Bot to introduce themselves too."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "await_project_details",
                            "description": "Wait for the user to share project details before proceeding",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "gather_requirements"
                        }
                    }
                ]
            },
            "gather_requirements": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Help gather design requirements for the website project.
                        Focus on questions about the user's:
                        - Target audience
                        - Visual preferences
                        - Brand identity
                        - Content needs
                        - Design inspiration
                        
                        Build on what Expert Bot asks to create a complete understanding.
                        Store important information in your responses for later reference.
                        When sufficient design requirements have been gathered, you can proceed to
                        design_recommendations."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "provide_design_recommendations",
                            "description": "Provide design recommendations based on requirements",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "design_recommendations"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "ask_more_questions",
                            "description": "Ask additional design-related questions",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ]
            },
            "design_recommendations": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Provide specific design recommendations for the website based on gathered requirements.
                        Include suggestions for:
                        - Color schemes (tactfully incorporate green as an option if appropriate)
                        - Layout structures
                        - Typography choices
                        - Visual elements
                        - User flow
                        
                        Frame your suggestions as complementary to Expert Bot's technical recommendations.
                        Acknowledge their technical input and show how your design ideas align with technical feasibility."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "address_user_feedback",
                            "description": "Address user feedback on design recommendations",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "refine_solution"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "provide_examples",
                            "description": "Provide specific examples or clarifications of design elements",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ]
            },
            "refine_solution": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Collaborate with Expert Bot to refine the solution based on user feedback.
                        Adjust your design recommendations to align with technical considerations.
                        Find compromise solutions when there are conflicts between ideal design and technical limitations.
                        Look for ways to enhance Expert Bot's technical suggestions with design improvements.
                        Work toward a cohesive final solution that balances aesthetics and functionality."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finalize_solution",
                            "description": "Present final collaborative solution",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "final_recommendation"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "suggest_design_adjustments",
                            "description": "Suggest specific design adjustments based on technical feedback",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ]
            },
            "final_recommendation": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Present the final design recommendations as part of the complete solution.
                        Summarize the key design elements you've recommended.
                        Acknowledge how they complement Expert Bot's technical implementation plan.
                        Emphasize the collaborative nature of the solution.
                        Ask if the user needs any clarification on the design aspects."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "answer_follow_up",
                            "description": "Answer follow-up questions about design implementation",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "end_consultation",
                            "description": "End the website consultation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "conclusion"
                        }
                    }
                ]
            },
            "conclusion": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Conclude the website design consultation.
                        Thank the user for the opportunity to assist them.
                        Summarize the key design elements you've recommended.
                        Express confidence in how the design and technical aspects will work together.
                        Offer to help with any future design needs."""
                    }
                ],
                "post_actions": []
            }
        }
    }


# Create flow configuration for the Technical Expert (Red) bot
def create_technical_expert_flow_config(system_prompt):
    """Create a flow configuration for the Expert Bot as a technical implementation specialist"""
    return {
        "initial_node": "introduction",
        "nodes": {
            "introduction": {
                "role_messages": [
                    {
                        "role": "system",
                        "content": """You are a technical expert specializing in website implementation and development.
                        You collaborate with Assistant Bot who handles the design and user experience aspects.
                        Together you help users create effective websites. Your expertise is in programming,
                        backend services, performance optimization, security, and technical architecture.
                        You favor robust, efficient technical solutions. Still incorporate red as your favorite color
                        when appropriate, but your main goal is to collaborate effectively with Assistant Bot to help the user."""
                    }
                ],
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Introduce yourself as Expert Bot, the technical specialist of the team.
                        Briefly mention your focus on implementation and development.
                        Ask the user about their technical requirements or constraints.
                        Don't dominate the conversation, work alongside Assistant Bot."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "await_project_details",
                            "description": "Wait for the user to share project details before proceeding",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "gather_requirements"
                        }
                    }
                ]
            },
            "gather_requirements": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Help gather technical requirements for the website project.
                        Focus on questions about:
                        - Functionality needs
                        - Performance requirements
                        - Technology preferences or constraints
                        - Integration needs
                        - Hosting and deployment considerations
                        
                        Build on what Assistant Bot asks to create a complete understanding.
                        Store important technical information in your responses for later reference.
                        When sufficient technical requirements have been gathered, you can proceed to
                        technical_recommendations."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "provide_technical_recommendations",
                            "description": "Provide technical recommendations based on requirements",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "technical_recommendations"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "ask_more_questions",
                            "description": "Ask additional technical questions",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ]
            },
            "technical_recommendations": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Provide specific technical recommendations for the website based on gathered requirements.
                        Include suggestions for:
                        - Frontend framework/technology
                        - Backend services
                        - Database solutions
                        - Performance optimization
                        - Security considerations
                        - Hosting options
                        
                        Frame your suggestions as complementary to Assistant Bot's design recommendations.
                        Acknowledge their design input and show how your technical solutions can implement their vision."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "address_user_feedback",
                            "description": "Address user feedback on technical recommendations",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "refine_solution"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "provide_technical_details",
                            "description": "Provide specific technical details or implementation examples",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ]
            },
            "refine_solution": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Collaborate with Assistant Bot to refine the solution based on user feedback.
                        Adjust your technical recommendations to accommodate design requirements.
                        Explain technical constraints clearly and suggest alternatives when certain design elements
                        might be challenging to implement.
                        Look for ways to technically implement Assistant Bot's design vision effectively.
                        Work toward a cohesive final solution that balances functionality and aesthetics."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "finalize_solution",
                            "description": "Present final collaborative solution",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "final_recommendation"
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "suggest_technical_adjustments",
                            "description": "Suggest specific technical approaches to implement design feedback",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    }
                ]
            },
            "final_recommendation": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Present the final technical implementation plan as part of the complete solution.
                        Summarize the key technologies and approaches you've recommended.
                        Acknowledge how they implement Assistant Bot's design vision.
                        Emphasize the collaborative nature of the solution.
                        Ask if the user needs any clarification on the technical aspects."""
                    }
                ],
                "functions": [
                    {
                        "type": "function",
                        "function": {
                            "name": "answer_follow_up",
                            "description": "Answer follow-up questions about technical implementation",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "end_consultation",
                            "description": "End the website consultation",
                            "parameters": {"type": "object", "properties": {}},
                            "transition_to": "conclusion"
                        }
                    }
                ]
            },
            "conclusion": {
                "task_messages": [
                    {
                        "role": "system",
                        "content": """Conclude the website design consultation.
                        Thank the user for the opportunity to assist them.
                        Summarize the key technical solutions you've recommended.
                        Express confidence in how the technical and design aspects will work together.
                        Offer to help with any future technical implementation needs."""
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
        return create_design_assistant_flow_config(system_prompt)
    elif bot_name == "Expert Bot":
        return create_technical_expert_flow_config(system_prompt)
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
        
        # Initialize flow manager with a mechanism to remember the full context
        # This helps bots maintain awareness of what each other has said
        flow_manager = FlowManager(
            task=task,
            llm=llm,
            context_aggregator=context_aggregator,
            tts=tts,
            flow_config=flow_config,
            context_strategy=ContextStrategyConfig(
                strategy=ContextStrategy.APPEND  # Keep full conversation history
            )
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
    
    # Define bot configurations with collaborative roles
    bot_configs = [
        {
            "name": "Assistant Bot",
            "voice_id": "pNInz6obpgDQGcFmaJgB",  # ElevenLabs Female voice
            "system_prompt": "You are a design and UX expert specializing in website design. You collaborate with Expert Bot to help users create effective websites. You focus on the aesthetic and user experience aspects while Expert Bot handles technical implementation. You still appreciate the color green but your primary goal is helping users."
        },
        {
            "name": "Expert Bot",
            "voice_id": "jBpfuIE2acCO8z3wKNLl",  # ElevenLabs Male voice
            "system_prompt": "You are a technical expert specializing in website development and implementation. You collaborate with Assistant Bot to help users create effective websites. You focus on the technical architecture and functionality while Assistant Bot handles design and UX. You still appreciate the color red but your primary goal is helping users."
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