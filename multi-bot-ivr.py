import asyncio
import os
import sys
import multiprocessing
import re
import time
from datetime import datetime, timedelta

import argparse
import json
import uuid

import aiohttp
from deepgram import LiveOptions
from dotenv import load_dotenv
from loguru import logger
from runner import configure


from pipecat.frames.frames import (
    BotInterruptionFrame,
    EndFrame,
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
from typing import Any, Dict

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.deepgram import DeepgramSTTService, Language, LiveOptions
from pipecat.transports.services.daily import DailyParams, DailyTransport


# Load environment variables in the main process
load_dotenv(override=True)

# Global variables for coordination
ROOM_URL = None
GOODBYE_EVENT = None  # Event to signal when goodbye was detected
CALL_START_TIME = None  # Track when the call starts
CALL_END_TIME = None  # Track when the call ends


# This must be a top-level function, not nested inside another function
def bot_process_runner(room_url, bot_name, voice_id, system_prompt, goodbye_event, call_stats):
    """Function to run in a separate process - must be at module level for pickling"""
    # Configure logging for this process
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
    
    # Load environment variables again in child process
    load_dotenv(override=True)
    
    # Run the bot
    asyncio.run(run_bot(room_url, bot_name, voice_id, system_prompt, goodbye_event, call_stats))

# Configure a single bot with its own pipeline
async def run_bot(room_url, bot_name, voice_id, system_prompt, goodbye_event, call_stats):
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
            
        @transport.event_handler("on_transcription_message")
        async def on_transcription_message(transport, message: Dict[str, Any]):
            participant_id = message.get("participantId")
            text = message.get("text")
            is_final = message["rawResponse"]["is_final"]
            logger.info(f"Transcription from {participant_id}: {text} (final: {is_final})")

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            # Record call start time when first participant joins
            if not call_stats["start_time"]:
                call_stats["start_time"] = time.time()
                logger.info(f"Call started at: {datetime.fromtimestamp(call_stats['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Kick off the conversation.
            messages.append({"role": "system", "content": f"Please introduce yourself as {bot_name} to the user."})
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            logger.info(f"Participant left: {participant.get('id')} - reason: {reason}")
            # Queue an EndFrame instead of canceling directly
            await task.queue_frames([EndFrame()])
            
        # Function to periodically write the conversation transcript to a file
        async def update_transcript():
            """Periodically write the conversation context to a file."""
            # Create a transcript directory if it doesn't exist
            os.makedirs("transcripts", exist_ok=True)
            
            # Create a filename based on the bot name and start time
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcripts/{bot_name}_{timestamp}.txt"
            
            while True:
                try:
                    # Get the current messages
                    messages = context.get_messages_for_persistent_storage()
                    
                    # Format the messages for better readability
                    formatted_messages = []
                    for msg in messages:
                        role = msg.get("role", "unknown")
                        content = msg.get("content", "")
                        formatted_messages.append(f"[{role.upper()}]: {content}\n")
                    
                    # Write to the file
                    with open(filename, "w") as f:
                        f.write(f"=== Transcript for {bot_name} ===\n")
                        f.write(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        f.writelines(formatted_messages)
                        
                    logger.debug(f"Updated transcript for {bot_name}")
                    
                    # Write a final transcript when the goodbye event is set
                    if goodbye_event.is_set():
                        # Create a final transcript with "final" in the name
                        final_filename = f"transcripts/{bot_name}_{timestamp}_final.txt"
                        with open(final_filename, "w") as f:
                            f.write(f"=== FINAL Transcript for {bot_name} ===\n")
                            f.write(f"Call ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                            f.writelines(formatted_messages)
                        logger.info(f"Saved final transcript for {bot_name} to {final_filename}")
                        break
                        
                except Exception as e:
                    logger.error(f"Error updating transcript for {bot_name}: {e}")
                
                # Wait before the next update
                await asyncio.sleep(5)  # Update every 5 seconds
            
        # Check the goodbye event and queue an EndFrame if set
        async def check_goodbye_event():
            while True:
                if goodbye_event.is_set():
                    logger.info(f"Bot {bot_name} detected goodbye event, sending EndFrame")
                    try:
                        # Record call end time when goodbye is detected
                        if not call_stats["end_time"]:
                            call_stats["end_time"] = time.time()
                            # Calculate and log call duration
                            duration = call_stats["end_time"] - call_stats["start_time"]
                            call_stats["duration"] = duration
                            logger.info(f"Call ended at: {datetime.fromtimestamp(call_stats['end_time']).strftime('%Y-%m-%d %H:%M:%S')}")
                            logger.info(f"Call duration: {timedelta(seconds=duration)}")
                        
                        # Queue the EndFrame to properly terminate the pipeline
                        await task.queue_frames([EndFrame()])
                    except Exception as e:
                        logger.error(f"Error queueing EndFrame: {e}")
                        # If we can't queue the frame, try to cancel directly
                        try:
                            await task.cancel()
                        except Exception as e:
                            logger.error(f"Error during task cancel: {e}")
                    break
                await asyncio.sleep(1)
                
        # Function to check for goodbye phrase in assistant responses
        async def check_for_goodbye():
            # Only check the last few messages to improve efficiency
            recent_messages = messages[-5:] if len(messages) > 5 else messages
            
            for msg in recent_messages:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Check for goodbye phrase
                    if content and "goodbye" in content.lower(): #and "thanks for calling" in content.lower():
                        logger.info(f"Bot {bot_name} said goodbye, signaling exit")
                        # Add a small delay before triggering exit to allow audio to finish
                        await asyncio.sleep(3)
                        goodbye_event.set()
                        return True
            return False
            
        # Monitor messages periodically in a background task
        async def monitor_for_goodbye():
            while True:
                await check_for_goodbye()
                await asyncio.sleep(1)
                
        # Also monitor for call timeout
        async def monitor_call_duration():
            max_call_duration = 300  # 5 minutes maximum call duration
            
            while True:
                if call_stats["start_time"] and not call_stats["end_time"]:
                    current_duration = time.time() - call_stats["start_time"]
                    
                    # Log duration every minute
                    if int(current_duration) % 60 == 0:
                        logger.info(f"Call in progress for: {timedelta(seconds=int(current_duration))}")
                    
                    # End call if exceeded maximum duration
                    if current_duration > max_call_duration:
                        logger.warning(f"Call exceeded maximum duration of {max_call_duration} seconds")
                        # Add timeout message to prompt the bot to say goodbye
                        messages.append({
                            "role": "system", 
                            "content": "The call has reached its maximum duration. Please politely end the conversation now by saying 'Thanks for calling American Express, goodbye!'"
                        })
                        await task.queue_frames([context_aggregator.user().get_context_frame()])
                
                await asyncio.sleep(1)
            
        # Start background tasks
        asyncio.create_task(check_goodbye_event())
        asyncio.create_task(monitor_for_goodbye())
        asyncio.create_task(monitor_call_duration())
        asyncio.create_task(update_transcript())  # Start the transcript update task

        runner = PipelineRunner()
        await runner.run(task)


# Function to start a bot in a separate process
def start_bot_process(room_url, bot_name, voice_id, system_prompt, goodbye_event, call_stats):
    """Start a bot in a separate process using the top-level function"""
    # Create and start the process with the top-level function
    process = multiprocessing.Process(
        target=bot_process_runner,
        args=(room_url, bot_name, voice_id, system_prompt, goodbye_event, call_stats)
    )
    process.start()
    return process

class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")

async def main():
    """Main function to coordinate bot processes"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run multi-bot conversation simulation")
    parser.add_argument("--test_task", type=str, default="current_balance", 
                        help="The task to test (e.g., current_balance, recent_transactions)")
    parser.add_argument("--test_id", type=str, default=str(uuid.uuid4()),
                        help="Unique identifier for this test run")
    parser.add_argument("--max_duration", type=int, default=300,
                        help="Maximum duration for the call in seconds")
    args = parser.parse_args()
    
    test_task = args.test_task
    test_id = args.test_id
    max_call_duration = args.max_duration
    
    # Configure logging
    logger.remove(0)
    logger.add(sys.stderr, level="DEBUG")
    
    # Get a single room URL that all bots will join
    async with aiohttp.ClientSession() as session:
        (room_url, _) = await configure(session)
        
    # Load IVR tree and mock data - moved from inline to modular approach
    ivr_tree = load_ivr_tree()
    mock_data = load_mock_data()

    # Create a shared event for signaling goodbye
    goodbye_event = multiprocessing.Event()
    
    # Create a Manager for sharing call stats between processes
    manager = multiprocessing.Manager()
    call_stats = manager.dict({
        "start_time": None,  # Will be set when first participant joins
        "end_time": None,    # Will be set when goodbye is detected
        "duration": None,    # Will be calculated at the end
        "test_task": test_task,
        "test_id": test_id
    })
    
    # Define bot configurations - customize as needed
    bot_configs = [
        {
            "name": "Jacob Smith",
            "voice_id": "pNInz6obpgDQGcFmaJgB",  # ElevenLabs Male voice
            "system_prompt": f"Your name is Jacob Smith and you are emulating a customer calling American Express to check on your {test_task}. Do not speak unless spoken to."
        },
        {
            "name": "Emma - American Express Customer Support Agent",
            "voice_id": "jBpfuIE2acCO8z3wKNLl",  # ElevenLabs Female voice
            "system_prompt": f"""You are Emma, an AI Agent that works at American Express. You help users with warm, friendly, and concise responses. 
            You can help with the following options:
            {ivr_tree}
            The customer calling you today is named Jacob Smith and has the following data. Refer to him strictly as Mr. Smith:
            {mock_data}
            This is the only data that exists for the user.
            Your responses will be transcribed to audio so do not use any special characters. i.e $19.99 = 19 dollars and 99 cents.
            Read all dollar amounts in word form.
            When the customer needs no further assistance, say 'Thanks for calling American Express, goodbye!'"""
        }
    ]
    
    # Create directories for results
    os.makedirs("transcripts", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Start each bot in a separate process
    processes = []
    for bot in bot_configs:
        logger.info(f"Starting bot: {bot['name']}")
        process = start_bot_process(
            room_url, 
            bot["name"], 
            bot["voice_id"], 
            bot["system_prompt"],
            goodbye_event,
            call_stats
        )
        processes.append(process)
        time.sleep(2)
    
    # Wait for KeyboardInterrupt or other signal to exit
    try:
        logger.info(f"Started {len(processes)} bots. Testing task: {test_task} with ID: {test_id}")
        # Wait for all processes to complete or for goodbye event
        max_wait_time = max_call_duration
        start_time = time.time()
        
        while not goodbye_event.is_set() and time.time() - start_time < max_wait_time:
            if all(not p.is_alive() for p in processes):
                break
            time.sleep(1)
            
        # If goodbye event was set, wait a short time for natural closure
        if goodbye_event.is_set():
            logger.info("Goodbye detected, waiting for processes to complete naturally...")
            time.sleep(5)  # Give processes time to exit naturally
            
        # Check if any processes are still running and terminate them
        for process in processes:
            if process.is_alive():
                logger.info(f"Terminating process {process.pid}")
                process.terminate()
                
    except KeyboardInterrupt:
        logger.info("Exiting due to keyboard interrupt")
        # Record end time if not already set
        if call_stats["start_time"] and not call_stats["end_time"]:
            call_stats["end_time"] = time.time()
            call_stats["duration"] = call_stats["end_time"] - call_stats["start_time"]
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
        
        # Print final call stats summary
        if call_stats["start_time"]:
            if not call_stats["end_time"]:
                call_stats["end_time"] = time.time()
                call_stats["duration"] = call_stats["end_time"] - call_stats["start_time"]
            
            logger.info("=== Call Statistics Summary ===")
            logger.info(f"Start time: {datetime.fromtimestamp(call_stats['start_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"End time: {datetime.fromtimestamp(call_stats['end_time']).strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"Total duration: {timedelta(seconds=call_stats['duration'])}")
            logger.info("============================")
        
        # After all processes have terminated, collect transcripts
        transcript_data = collect_transcripts("transcripts")
        
        # Create result JSON file
        result_data = {
            "test_task": test_task,
            "test_id": test_id,
            "start_time": datetime.fromtimestamp(call_stats["start_time"]).strftime('%Y-%m-%d %H:%M:%S') if call_stats["start_time"] else None,
            "end_time": datetime.fromtimestamp(call_stats["end_time"]).strftime('%Y-%m-%d %H:%M:%S') if call_stats["end_time"] else None,
            "duration_seconds": call_stats["duration"],
            "duration_formatted": str(timedelta(seconds=call_stats["duration"])) if call_stats["duration"] else None,
            "transcripts": transcript_data
        }
        
        # Write results to JSON file
        result_filename = f"results/test_{test_task}_{test_id}.json"
        with open(result_filename, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        logger.info(f"Test results saved to {result_filename}")


# Add these new helper functions before the main function

def load_ivr_tree():
    """Load IVR tree from file or return default"""
    # You can modify this to load from a JSON file if needed
    return """
{
  "Main Menu": {
    "1. Account Information": {
      "1.1. Check Balance": {
        "1.1.1. Current Balance": {},
        "1.1.2. Available Credit": {},
        "1.1.3. Last Statement Balance": {}
      },
      "1.2. Recent Transactions": {
        "1.2.1. Last 5 Transactions": {},
        "1.2.2. Transactions Over $500": {},
        "1.2.3. Specific Date Range": {}
      },
      "1.3. Payment Due Date": {
        "1.3.1. Current Payment Due Date": {},
        "1.3.2. Minimum Payment": {},
        "1.3.3. Payment History": {}
      }
    },
    "2. Rewards": {
      "2.1. Rewards Balance": {
        "2.1.1. Points Balance": {},
        "2.1.2. Cashback Balance": {},
        "2.1.3. Expiring Rewards": {}
      },
      "2.2. Redeem Rewards": {
        "2.2.1. Redeem for Travel": {},
        "2.2.2. Redeem for Gift Cards": {},
        "2.2.3. Redeem for Statement Credit": {}
      },
      "2.3. Explore Offers": {
        "2.3.1. Travel Offers": {},
        "2.3.2. Dining Offers": {},
        "2.3.3. Shopping Offers": {}
      }
    },
    "3. Lost or Stolen Card": {
      "3.1. Report Lost Card": {
        "3.1.1. Block Current Card": {},
        "3.1.2. Issue Replacement Card": {},
        "3.1.3. Check Card Status": {}
      },
      "3.2. Request Replacement": {
        "3.2.1. Standard Delivery": {},
        "3.2.2. Expedited Delivery": {},
        "3.2.3. Digital Card Activation": {}
      },
      "3.3. Freeze Account": {
        "3.3.1. Temporary Freeze": {},
        "3.3.2. Permanent Freeze": {},
        "3.3.3. Freeze History": {}
      }
    },
    "4. Customer Support": {
      "4.1. Speak to an Agent": {
        "4.1.1. Billing Queries": {},
        "4.1.2. Technical Assistance": {},
        "4.1.3. General Inquiries": {}
      },
      "4.2. FAQs": {
        "4.2.1. Billing FAQs": {},
        "4.2.2. Rewards FAQs": {},
        "4.2.3. Security FAQs": {}
      },
      "4.3. Feedback Submission": {
        "4.3.1. Rate Service": {},
        "4.3.2. Submit Complaint": {},
        "4.3.3. Share Suggestions": {}
      }
    },
    "5. Technical Support": {
      "5.1. Mobile App Assistance": {
        "5.1.1. App Installation Issues": {},
        "5.1.2. Login Problems": {},
        "5.1.3. Payment Processing Errors": {}
      },
      "5.2. Online Banking Issues": {
        "5.2.1. Forgotten Password": {},
        "5.2.2. Account Lockout": {},
        "5.2.3. Transaction Failures": {}
      },
      "5.3. Card Activation Problems": {
        "5.3.1. Activation Code Issues": {},
        "5.3.2. Invalid Card Details": {},
        "5.3.3. Activation Delays": {}
      }
    }
  }
}

"""

def load_mock_data():
    """Load mock data from file or return default"""
    # You can modify this to load from a JSON file if needed
    return {
    "account_balance": {
        "current_balance": "2 thousand 5 hundred and 47 dollars and 89 cents",
        "available_credit": "7 thousand 4 hundred and 52 dollars and 11 cents",
        "last_statement_balance": "3 thousand 2 hundred and 14 dollars and 56 cents"
    },
    "recent_transactions": [
        {"date": "2025-03-05", "description": "AMAZON.COM", "amount": "156 dollars and 78 cents"},
        {"date": "2025-03-04", "description": "STARBUCKS", "amount": "12 dollars and 45 cents"},
        {"date": "2025-03-03", "description": "WALMART", "amount": "89 dollars and 32 cents"},
        {"date": "2025-03-01", "description": "NETFLIX", "amount": "19 dollars and 99 cents"},
        {"date": "2025-02-28", "description": "UBER", "amount": "34 dollars and 56 cents"}
    ],
    "large_transactions": [
        {"date": "2025-02-25", "description": "BEST BUY", "amount": "8 hundred and 99 dollars and 99 cents"},
        {"date": "2025-02-15", "description": "DELTA AIRLINES", "amount": "7 hundred and 52 dollars and 34 cents"}
    ],
    "payment_info": {
        "due_date": "2025-03-25",
        "minimum_payment": "85 dollars and 00 cents",
        "payment_history": ["2025-02-15", "2025-01-15", "2024-12-15"]
    },
    "rewards": {
        "points_balance": "15 thousand 3 hundred and 20 points",
        "cashback_balance": "120 dollars and 50 cents",
        "expiring_rewards": "2 thousand points expiring on 2025-04-01"
    },
    "redeem_rewards": {
        "travel": "Use 20 thousand points for a flight to New York",
        "gift_cards": "Redeem 5 thousand points for a 50-dollar Amazon gift card",
        "statement_credit": "Use 10 thousand points for a 100-dollar statement credit"
    },
    "offers": {
        "travel": "10% off select hotels with your card",
        "dining": "Earn 5x points on restaurant purchases this month",
        "shopping": "Get 15% cashback on select online retailers"
    },
    "card_management": {
        "lost_or_stolen": {
            "block_card": "Your card has been blocked successfully.",
            "issue_replacement": "A new card will be delivered within 5-7 business days.",
            "check_status": "Your replacement card is in transit and will arrive on 2025-03-10."
        },
        "freeze_account": {
            "temporary_freeze": "Your account has been temporarily frozen.",
            "permanent_freeze": "Your account is now permanently frozen. Contact support to reactivate.",
            "freeze_history": ["2024-12-10", "2023-07-22"]
        }
    },
    "customer_support": {
        "agent": {
            "billing_queries": "Transferring you to a billing specialist...",
            "technical_assistance": "Connecting to a technical support agent...",
            "general_inquiries": "Please hold for the next available representative."
        },
        "faqs": {
            "billing": "Your statement is generated on the 5th of every month.",
            "rewards": "Points expire after 24 months of inactivity.",
            "security": "Enable two-factor authentication for added security."
        },
        "feedback": {
            "rate_service": "Thank you for your feedback!",
            "submit_complaint": "Your complaint has been recorded.",
            "share_suggestions": "Your suggestion has been submitted."
        }
    },
    "technical_support": {
        "mobile_app": {
            "installation_issues": "Ensure you have at least 500MB free storage.",
            "login_problems": "Reset your password via the app settings.",
            "payment_errors": "Check if your card is active and has sufficient balance."
        },
        "online_banking": {
            "forgotten_password": "Reset your password through the website.",
            "account_lockout": "Contact support to unlock your account.",
            "transaction_failures": "Ensure your card details are correct."
        },
        "card_activation": {
            "activation_code": "Enter the activation code from your welcome letter.",
            "invalid_details": "Double-check your card number and expiration date.",
            "activation_delays": "Your activation request is being processed."
        }
    }
}


def collect_transcripts(transcript_dir):
    """Collect all transcript data from files"""
    transcript_data = {}
    
    if os.path.exists(transcript_dir):
        # Find all final transcript files
        final_transcripts = [f for f in os.listdir(transcript_dir) if "final" in f]
        
        if final_transcripts:
            for transcript_file in final_transcripts:
                file_path = os.path.join(transcript_dir, transcript_file)
                bot_name = transcript_file.split('_')[0]  # Extract bot name from filename
                
                try:
                    with open(file_path, 'r') as f:
                        transcript_content = f.read()
                        transcript_data[bot_name] = transcript_content
                    os.remove(file_path)
                except Exception as e:
                    logger.error(f"Error reading transcript file {file_path}: {e}")
        else:
            # If no final transcripts, use the latest regular transcripts
            all_transcripts = os.listdir(transcript_dir)
            bot_names = set([f.split('_')[0] for f in all_transcripts])
            
            for bot_name in bot_names:
                bot_files = [f for f in all_transcripts if f.startswith(bot_name)]
                if bot_files:
                    # Get the most recent file for this bot
                    latest_file = max(bot_files, key=lambda f: os.path.getctime(os.path.join(transcript_dir, f)))
                    file_path = os.path.join(transcript_dir, latest_file)
                    
                    try:
                        with open(file_path, 'r') as f:
                            transcript_content = f.read()
                            transcript_data[bot_name] = transcript_content
                    except Exception as e:
                        logger.error(f"Error reading transcript file {file_path}: {e}")
    
    return transcript_data

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')  # More reliable cross-platform support
    asyncio.run(main())