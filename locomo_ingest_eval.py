#!/usr/bin/env python3
"""
LoCoMo-MC10 Dataset Test
Tests long conversation memory across multiple threads with the Backboard API.

The LoCoMo-MC10 dataset tests 5 conversation memory abilities:
- Single-hop reasoning (SH)
- Multi-hop reasoning (MH)
- Temporal reasoning (TR)
- Open-domain knowledge (OD)

Configuration:
- Processes all conversations in the dataset from start to finish
- Creates a new assistant for each conversation (isolated memory per conversation)
- Verbose logging enabled (shows all conversation turns streaming)
- Evaluates questions from categories 1-4 only (excludes category 5 - adversarial questions)

Dry Run Mode:
- Set DRY_RUN = True to simulate execution without making actual API calls
- Useful for testing the flow, estimating time, and verifying data structure
- Shows what would happen without consuming API credits or time
"""

import asyncio
import json
import uuid
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict
from datetime import datetime
import httpx
import numpy as np
from dotenv import load_dotenv
import os
from openai import AsyncOpenAI

# Load environment variables from .env file
load_dotenv()

# Note: We load from local locomo_dataset.json file only (no remote downloads)

# Configuration - Load from environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://app.backboard.io/api")
API_KEY = os.getenv("BACKBOARD_API_KEY")
if not API_KEY:
    raise ValueError("BACKBOARD_API_KEY environment variable is required. Please set it in your .env file.")

TIMEOUT = 300.0  # Extended timeout for network stability

HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}


# Test configuration - Simple and clean
VERBOSE_LOGGING = True  # Show all conversation turns streaming
CREATE_ASSISTANT_PER_CONVERSATION = True  # Create a new assistant for each conversation (isolated memory)
DRY_RUN = True  # Set to True to simulate the run without making actual API calls


def convert_to_iso8601(timestamp_str: str) -> Optional[str]:
    """
    Convert human-readable timestamps to ISO 8601 format.
    
    Examples:
        "1:56 pm on 8 May, 2023" -> "2023-05-08T13:56:00Z"
        "10:30 am on 15 January, 2023" -> "2023-01-15T10:30:00Z"
    
    Returns None if conversion fails.
    """
    if not timestamp_str:
        return None
    
    try:
        # Parse format like "1:56 pm on 8 May, 2023"
        # Split by " on " to separate time and date
        parts = timestamp_str.lower().strip().split(" on ")
        if len(parts) != 2:
            return None
        
        time_part = parts[0].strip()  # "1:56 pm"
        date_part = parts[1].strip()  # "8 May, 2023" or "8 may 2023"
        
        # Remove comma if present
        date_part = date_part.replace(',', '')
        
        # Parse the datetime
        datetime_str = f"{date_part} {time_part}"
        
        # Try different date formats
        for fmt in [
            "%d %B %Y %I:%M %p",      # 8 May 2023 1:56 pm
            "%d %b %Y %I:%M %p",      # 8 May 2023 1:56 pm (abbreviated month)
            "%B %d %Y %I:%M %p",      # May 8 2023 1:56 pm
            "%b %d %Y %I:%M %p",      # May 8 2023 1:56 pm (abbreviated)
        ]:
            try:
                dt = datetime.strptime(datetime_str, fmt)
                # Convert to ISO 8601 format with UTC timezone
                return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
            except ValueError:
                continue
        
        return None
        
    except Exception as e:
        print(f"      WARNING: Failed to convert timestamp '{timestamp_str}': {e}")
        return None


async def evaluate_answer_with_llm(
    openai_client: AsyncOpenAI,
    question: str,
    expected_answer: str,
    ai_response: str,
    question_type: str
) -> Dict[str, Any]:
    """
    Use GPT-4o-mini as a judge to evaluate if the AI's answer is correct.
    Uses the updated accuracy prompt for conversational memory evaluation.
    
    Returns a dict with:
        - is_correct: bool (True if answer is correct)
        - reasoning: str (brief explanation)
        - error: str (if evaluation failed)
    """
    
    # Updated accuracy prompt for conversational memory evaluation
    ACCURACY_PROMPT = f"""
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user), 
    (2) a 'gold' (ground truth) answer, 
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. 

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {expected_answer}
Generated answer: {ai_response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. 
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Return your response in JSON format with two keys: "reasoning" for your explanation and "label" for CORRECT or WRONG.
"""

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4.1",  
            messages=[
                {"role": "system", "content": "You are evaluating conversational AI memory recall. Return JSON only with the format requested."},
                {"role": "user", "content": ACCURACY_PROMPT}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent evaluation
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Handle the new format - convert "CORRECT"/"WRONG" to boolean
        label = result.get("label", "WRONG").upper()
        is_correct = label == "CORRECT"
        
        return {
            "is_correct": is_correct,
            "reasoning": result.get("reasoning", "No reasoning provided"),
            "error": None
        }
        
    except Exception as e:
        print(f"      WARNING: LLM judge evaluation failed: {e}")
        return {
            "is_correct": False,
            "reasoning": "",
            "error": str(e)
        }


class LoCoMoTester:
    """Test runner for LoCoMo-MC10 dataset with Backboard API"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "X-API-Key": api_key
        }
        self.assistant_id = None
        
        # Initialize OpenAI client for LLM-as-judge evaluation
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("WARNING: OPENAI_API_KEY not found in environment. LLM judge evaluation will be disabled.")
            self.openai_client = None
        else:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        
        # Overall results across all conversations
        self.results = {
            "total": 0,
            "total_evaluated": 0,
            "total_correct": 0,
            "by_type": defaultdict(lambda: {
                "total": 0, 
                "responses": [], 
                "correct_count": 0,
                "evaluated_count": 0
            }),
            "questions": [],
            "all_responses": []
        }
        
        # Per-conversation results for tracking individual conversation accuracy
        self.conversation_results = []  # List of results per conversation
        
    async def create_assistant(self, client: httpx.AsyncClient) -> str:
        """Create a new assistant with memory enabled"""
        print("Creating new assistant with memory enabled...")
        
        if DRY_RUN:
            self.assistant_id = f"dry_run_assistant_{str(uuid.uuid4())[:8]}"
            print(f"[DRY RUN] Would create assistant: LoCoMo Test Assistant ({self.assistant_id})")
            return self.assistant_id
        
        # Assistant instructions for concise answers
        instructions = (
            "You are a conversation memory assistant."
        )
        
        assistant_payload = {
            "name": f"LoCoMo Test Assistant {str(uuid.uuid4())[:8]}",
            "description": instructions,
            "llm_provider": "google",
            "llm_model_name": "gemini-2.5-pro",
            "tools": []
        }
        
        response = await client.post(
            f"{self.base_url}/assistants",
            headers=self.headers,
            json=assistant_payload,
            timeout=TIMEOUT
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create assistant: {response.status_code} - {response.text}")
        
        assistant_data = response.json()
        self.assistant_id = assistant_data.get("assistant_id")
        print(f"Created assistant: {assistant_data.get('name')} ({self.assistant_id})")
        return self.assistant_id
    
    async def create_thread(self, client: httpx.AsyncClient) -> str:
        """Create a new thread"""
        if DRY_RUN:
            thread_id = f"dry_run_thread_{str(uuid.uuid4())[:8]}"
            return thread_id
        
        response = await client.post(
            f"{self.base_url}/assistants/{self.assistant_id}/threads",
            headers=self.headers,
            json={},
            timeout=TIMEOUT
        )
        
        if response.status_code not in [200, 201]:
            raise Exception(f"Failed to create thread: {response.status_code} - {response.text}")
        
        thread_data = response.json()
        return thread_data.get("thread_id")
    
    async def send_message(self, client: httpx.AsyncClient, thread_id: str, 
                          content: str, memory_enabled: bool = True, 
                          show_streaming: bool = False, metadata: Optional[Dict] = None,
                          send_to_llm: bool = True) -> Dict[str, Any]:
        """Send a message to a thread with optional streaming display and metadata"""
        
        if DRY_RUN:
            # Simulate message sending
            if show_streaming:
                print(f"      [DRY RUN] User: {content[:100]}...")
                if send_to_llm:
                    print(f"      [DRY RUN] AI: [Simulated response]")
                else:
                    print(f"      [DRY RUN] [Context saved without LLM response]")
            return {
                "content": "[DRY RUN] Simulated AI response",
                "memory_operation_id": f"dry_run_mem_op_{str(uuid.uuid4())[:8]}",
                "retrieved_memories": []
            }
        
        if show_streaming:
            # Use streaming to show AI response in real-time
            return await self.send_message_streaming(client, thread_id, content, memory_enabled, metadata, send_to_llm)
        else:
            # Non-streaming (faster for loading conversations)
            form_data = {
                "content": content,
                "stream": "false",
                "memory": "auto" if memory_enabled else "off",
                "send_to_llm": "true" if send_to_llm else "false",
                "llm_provider": "google",
                "model_name": "gemini-2.5-pro",
            }
            
            # Add metadata if provided
            if metadata:
                form_data["metadata"] = json.dumps(metadata)
            
            # For form data, don't include Content-Type header
            form_headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
            
            response = await client.post(
                f"{self.base_url}/threads/{thread_id}/messages",
                headers=form_headers,
                data=form_data,
                timeout=TIMEOUT
            )
            
            if response.status_code != 200:
                raise Exception(f"Failed to send message: {response.status_code} - {response.text}")
            
            data = response.json()
            return {
                "content": data.get("content", ""),
                "memory_operation_id": data.get("memory_operation_id"),
                "retrieved_memories": data.get("retrieved_memories", [])
            }
    
    async def send_message_streaming(self, client: httpx.AsyncClient, thread_id: str,
                                     content: str, memory_enabled: bool = True, 
                                     metadata: Optional[Dict] = None,
                                     send_to_llm: bool = True) -> Dict[str, Any]:
        """Send a message and stream the response with optional metadata"""
        print(f"      User: {content}")
        
        # Show custom timestamp if provided (condensed for verbose mode)
        if metadata and 'custom_timestamp' in metadata:
            print(f"      Time: {metadata['custom_timestamp']}")
        
        # Only show "AI: " prefix if we're actually sending to LLM
        if send_to_llm:
            print(f"      AI: ", end="", flush=True)
        
        form_data = {
            "content": content,
            "stream": "true",
            "memory": "auto" if memory_enabled else "off",
            "send_to_llm": "true" if send_to_llm else "false"
        }
        
        # Add metadata if provided
        if metadata:
            form_data["metadata"] = json.dumps(metadata)
        
        form_headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        
        content_chunks = []
        memory_operation_id = None
        retrieved_memories = []
        
        async with client.stream(
            "POST",
            f"{self.base_url}/threads/{thread_id}/messages",
            headers=form_headers,
            data=form_data,
            timeout=TIMEOUT
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise Exception(f"Streaming failed: {response.status_code} - {error_text.decode()}")
            
            async for line in response.aiter_lines():
                if line.strip() == "":
                    continue
                
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")
                        
                        if event_type == "content_streaming":
                            content_chunk = data.get("content", "")
                            content_chunks.append(content_chunk)
                            print(content_chunk, end="", flush=True)
                        
                        elif event_type == "memory_retrieved":
                            retrieved_memories = data.get("memories", [])
                        
                        elif event_type == "message_saved":
                            # When send_to_llm=False, we get message_saved event with memory_operation_id
                            memory_operation_id = data.get("memory_operation_id")
                            print(f"      [Context saved without LLM response]")
                            break
                        
                        elif event_type == "run_ended":
                            memory_operation_id = data.get("memory_operation_id")
                            if not retrieved_memories:
                                retrieved_memories = data.get("retrieved_memories", [])
                            break
                    
                    except json.JSONDecodeError:
                        continue
        
        full_content = "".join(content_chunks)
        if send_to_llm:
            print()  # New line after streaming (only if we were streaming AI response)
        
        return {
            "content": full_content,
            "memory_operation_id": memory_operation_id,
            "retrieved_memories": retrieved_memories
        }
    
    async def wait_for_memory_operation(self, client: httpx.AsyncClient, 
                                       operation_id: str, 
                                       timeout_seconds: int = 60,
                                       verbose: bool = False) -> Optional[Dict]:
        """Wait for memory operation to complete and show tracking details"""
        if not operation_id:
            if verbose:
                print("      WARNING: No memory operation id returned")
            return None
        
        if DRY_RUN:
            # Simulate instant completion in dry run
            if verbose:
                print(f"      [DRY RUN] Memory operation completed instantly")
            return {
                "status": "COMPLETED",
                "memory_ids": [
                    {"id": "dry_run_mem_1", "event": "ADD", "memory": "Simulated memory 1"}
                ]
            }
        
        start = time.time()
        while True:
            try:
                resp = await client.get(
                    f"{self.base_url}/assistants/memories/operations/{operation_id}",
                    headers=self.headers,
                    timeout=10.0
                )
                
                if resp.status_code == 200:
                    data = resp.json()
                    status = data.get("status")
                    memory_ids = data.get('memory_ids')
                    
                    # Check completion first (for speed)
                    if status in ("COMPLETED", "ERROR", "completed", "error"):
                        if verbose:
                            # Show final status with memory details
                            if memory_ids and isinstance(memory_ids, list) and len(memory_ids) > 0:
                                if isinstance(memory_ids[0], dict):
                                    # New format with per-memory operations
                                    print(f"      • status={status}, {len(memory_ids)} memories:")
                                    for mem in memory_ids:
                                        mem_id = mem.get('id')
                                        event = mem.get('event', 'UNKNOWN').upper()
                                        mem_text = mem.get('memory', '')
                                        # For NONE events, there's no ID (expected behavior)
                                        id_display = mem_id if mem_id else "N/A" if event == "NONE" else "unknown"
                                        print(f"         - [{event}] {id_display} | {mem_text}")
                                else:
                                    # Legacy format (simple list of IDs)
                                    print(f"      • status={status} memory_ids={memory_ids}")
                            else:
                                print(f"      • status={status}, no memories")
                            print(f"      Memory operation complete")
                        return data
                    
                    # Only show polling progress in verbose mode and not too often
                    if verbose and int(time.time() - start) % 3 == 0:  # Every 3 seconds
                        print(f"      Waiting: status={status}...")
                
            except Exception as e:
                if verbose:
                    print(f"      WARNING: Memory operation check error: {type(e).__name__}: {str(e)}")
            
            if time.time() - start > timeout_seconds:
                if verbose:
                    print(f"      TIMEOUT: Waiting for memory operation")
                return None
            
            # Fast polling (0.2s) for first 10 seconds, then slow down to 1s
            if time.time() - start < 10:
                await asyncio.sleep(0.2)
            else:
                await asyncio.sleep(1.0)
    
    async def load_conversation_sessions(self, client: httpx.AsyncClient,
                                        sessions: List[List], 
                                        session_summaries: List[str],
                                        session_datetimes: List[str],
                                        verbose: bool = True) -> List[str]:
        """
        Load conversation sessions into separate threads.
        Each session gets its own thread to simulate multi-session conversations.
        Returns list of thread IDs.
        """
        thread_ids = []
        
        for session_idx, (session, summary, datetime_str) in enumerate(
            zip(sessions, session_summaries, session_datetimes), 1
        ):
            # Create thread for this session
            print(f"\n   Session {session_idx}/{len(sessions)}")
            thread_id = await self.create_thread(client)
            print(f"      Thread: {thread_id}")
            
            thread_ids.append(thread_id)
            
            if datetime_str:
                print(f"      Session datetime (original): {datetime_str}")
            
            # Prepare metadata with custom timestamp if available
            metadata = None
            if datetime_str:
                # Convert to ISO 8601 format
                iso_timestamp = convert_to_iso8601(datetime_str)
                if iso_timestamp:
                    print(f"      Session datetime (ISO 8601): {iso_timestamp}")
                    metadata = {"custom_timestamp": iso_timestamp}
                else:
                    print(f"      WARNING: Could not convert timestamp, skipping metadata")
            
            print(f"      Loading {len(session)} turns", end="", flush=True)
            
            # Load conversation turns from this session
            for turn_idx, turn in enumerate(session, 1):
                
                # All turns should be dicts now (normalized in main())
                if not isinstance(turn, dict):
                    continue
                
                speaker = turn.get("speaker", "Unknown")
                text = turn.get("text", "")
                
                if not text:
                    continue
                
                # Check if this turn has an image - if so, add query and BLIP caption to the message
                if "img_url" in turn and turn["img_url"]:
                    query = turn.get("query", "")
                    blip_caption = turn.get("blip_caption", "")
                    
                    if query or blip_caption:
                        # Format image information clearly for fact extraction
                        # Query = what person searched for/intended to show (their perspective/description)
                        # Caption = what is actually visible in the image (literal visual content)
                        if query and blip_caption:
                            # Both available - make it clear these are two separate pieces of information
                            text = f"{text} [Sharing image - query: {query}. The image shows: {blip_caption}]"
                        elif query:
                            text = f"{text} [Sharing image - query for: {query}]"
                        elif blip_caption:
                            text = f"{text} [Sharing image that shows: {blip_caption}]"
                        
                        if verbose:
                            print(f"         🖼️  Image detected, using query and BLIP caption")
                
                message_content = f"{speaker}: {text}"
                
                # Retry logic for network errors
                max_retries = 3
                retry_delay = 2 if not DRY_RUN else 0  # No delay in dry run
                result = None
                
                for attempt in range(max_retries):
                    try:
                        if verbose:
                            print(f"\n      Turn {turn_idx}/{len(session)}")
                            # Show streaming for conversation turns - but DON'T send to LLM (just save context)
                            result = await self.send_message(client, thread_id, message_content, 
                                                            memory_enabled=True, show_streaming=True,
                                                            metadata=metadata, send_to_llm=False)
                        else:
                            # Non-streaming for faster loading - DON'T send to LLM (just save context)
                            result = await self.send_message(client, thread_id, message_content, 
                                                            memory_enabled=True, show_streaming=False,
                                                            metadata=metadata, send_to_llm=False)
                            if turn_idx % 10 == 0:  # Progress indicator
                                print(".", end="", flush=True)
                        break  # Success, exit retry loop
                    
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Check if it's a retryable network error
                        if "peer closed" in error_msg or "incomplete chunked" in error_msg or "connection" in error_msg:
                            if attempt < max_retries - 1:
                                if verbose:
                                    print(f"      ⚠️  Network error (attempt {attempt + 1}/{max_retries}): {e}")
                                    if not DRY_RUN:
                                        print(f"      ⏳ Retrying in {retry_delay}s...")
                                if retry_delay > 0:
                                    await asyncio.sleep(retry_delay)
                                continue
                            else:
                                if verbose:
                                    print(f"      ❌ Failed after {max_retries} attempts: {e}")
                                raise  # Re-raise after all retries exhausted
                        else:
                            # Non-retryable error, raise immediately
                            raise
                
                # Wait for memory operation to complete (only if one was created)
                # Note: With send_to_llm=false, there may be no memory operation
                if result and result.get("memory_operation_id"):
                    mem_op_id = result["memory_operation_id"]
                    if verbose:
                        print(f"      💾 Memory operation: {mem_op_id}")
                    mem_result = await self.wait_for_memory_operation(client, mem_op_id, 
                                                        timeout_seconds=10, verbose=verbose)
                    if not mem_result and verbose:
                        print(f"      ⚠️  Memory operation did not complete in time")
            
            print(f" Done")
        
        return thread_ids
    
    async def ask_question(self, client: httpx.AsyncClient, thread_id: str,
                          question: str) -> Dict[str, Any]:
        """Ask a question in a thread and return the answer with streaming (no prompt display)"""
        
        if DRY_RUN:
            print(f"\n   [DRY RUN] AI Response: This is a simulated answer to the question.")
            return {
                "content": "[DRY RUN] Simulated answer based on retrieved memories",
                "memory_operation_id": f"dry_run_qa_mem_op_{str(uuid.uuid4())[:8]}",
                "retrieved_memories": [
                    {"id": "dry_run_mem_1", "content": "Simulated retrieved memory 1"},
                    {"id": "dry_run_mem_2", "content": "Simulated retrieved memory 2"}
                ]
            }
        
        print(f"\n   AI Response:", end=" ", flush=True)
        
        # Use streaming for QA questions - SEND TO LLM (this is where we want actual responses)
        form_data = {
            "content": question,
            "stream": "true",
            "memory": "auto",
            "send_to_llm": "true"  # QA questions should get LLM responses
        }
        
        form_headers = {k: v for k, v in self.headers.items() if k.lower() != "content-type"}
        
        content_chunks = []
        memory_operation_id = None
        retrieved_memories = []
        
        async with client.stream(
            "POST",
            f"{self.base_url}/threads/{thread_id}/messages",
            headers=form_headers,
            data=form_data,
            timeout=TIMEOUT
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise Exception(f"Streaming failed: {response.status_code} - {error_text.decode()}")
            
            async for line in response.aiter_lines():
                if line.strip() == "":
                    continue
                
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")
                        
                        if event_type == "content_streaming":
                            content_chunk = data.get("content", "")
                            content_chunks.append(content_chunk)
                            print(content_chunk, end="", flush=True)
                        
                        elif event_type == "memory_retrieved":
                            retrieved_memories = data.get("memories", [])
                        
                        elif event_type == "run_ended":
                            memory_operation_id = data.get("memory_operation_id")
                            if not retrieved_memories:
                                retrieved_memories = data.get("retrieved_memories", [])
                            break
                    
                    except json.JSONDecodeError:
                        continue
        
        full_content = "".join(content_chunks)
        print()  # New line after streaming
        
        # Show retrieved memories if any
        if retrieved_memories:
            print(f"\n   Retrieved {len(retrieved_memories)} memories:")
            for idx, memory in enumerate(retrieved_memories, 1):
                # Handle both dict and string formats
                if isinstance(memory, dict):
                    memory_content = memory.get('content') or memory.get('memory') or memory.get('text') or str(memory)
                    memory_id = memory.get('id') or memory.get('memory_id') or 'N/A'
                    print(f"      [{idx}] ID: {memory_id}")
                    print(f"          Content: {memory_content}")
                else:
                    print(f"      [{idx}] {memory}")
        
        # Wait for memory operation
        if memory_operation_id:
            print(f"\n   Memory operation: {memory_operation_id}")
            await self.wait_for_memory_operation(client, memory_operation_id, 
                                                 timeout_seconds=10, verbose=True)
        
        return {
            "content": full_content,
            "memory_operation_id": memory_operation_id,
            "retrieved_memories": retrieved_memories
        }
    
    async def test_question_batch(self, client: httpx.AsyncClient, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Test all questions from a single conversation (with multiple QA items)"""
        conversation_id = item.get("sample_id", "unknown")
        all_qa_items = item.get("qa", [])
        
        # Filter out category 5 (adversarial questions)
        qa_items = [qa for qa in all_qa_items if qa.get("category") != 5]
        
        sessions = item.get("haystack_sessions", [])
        session_summaries = item.get("haystack_session_summaries", [])
        session_datetimes = item.get("haystack_session_datetimes", [])
        
        print(f"ID: {conversation_id}")
        print(f"Sessions: {len(sessions)} | Questions: {len(qa_items)} (filtered from {len(all_qa_items)}, excluding category 5)")
        
        try:
            thread_ids = []
            
            # Load conversation sessions into separate threads
            print(f"\nLoading {len(sessions)} conversation sessions...")
            thread_ids = await self.load_conversation_sessions(
                client, sessions, session_summaries, session_datetimes, verbose=VERBOSE_LOGGING
            )
            
            # Create a single new thread for asking all questions
            question_thread_id = await self.create_thread(client)
            print(f"\nQuestion Thread: {question_thread_id}")
            
            results = []
            
            # Track running accuracy for this batch
            batch_evaluated_count = 0
            batch_correct_count = 0
            
            # Ask all questions in the same thread
            for qa_idx, qa_item in enumerate(qa_items, 1):
                try:
                    question = qa_item.get("question", "")
                    category = qa_item.get("category", 0)
                    answer_text = qa_item.get("answer", "")
                    
                    # Map category to question type
                    category_map = {
                        1: 'single_hop',
                        2: 'temporal_reasoning',
                        3: 'multi_hop',
                        4: 'open_domain',
                        5: 'adversarial'
                    }
                    question_type = category_map.get(category, 'unknown')
                    
                    print(f"\n   {'='*70}")
                    print(f"   Question {qa_idx}/{len(qa_items)} [{question_type}]")
                    print(f"   {question}")
                    
                    # Ask the question directly (assistant has concise instructions already)
                    start_time = time.time()
                    result_data = await self.ask_question(client, question_thread_id, question)
                    response_time = 0.01 if DRY_RUN else (time.time() - start_time)  # Instant in dry run
                    ai_response = result_data.get("content", "")
                    
                    # Display results
                    print(f"\n   Expected Answer: {answer_text}")
                    print(f"   Response Time: {response_time:.2f}s")
                    
                    # Evaluate with LLM judge
                    evaluation = None
                    if DRY_RUN:
                        # Simulate evaluation
                        import random
                        is_correct = random.choice([True, False])
                        evaluation = {
                            "is_correct": is_correct,
                            "reasoning": "[DRY RUN] Simulated evaluation reasoning",
                            "error": None
                        }
                        print(f"   [DRY RUN] Evaluating with LLM judge... {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                    elif self.openai_client and ai_response:
                        print(f"   Evaluating with LLM judge...", end=" ", flush=True)
                        evaluation = await evaluate_answer_with_llm(
                            self.openai_client,
                            question,
                            str(answer_text),
                            ai_response,
                            question_type
                        )
                        
                        if evaluation and not evaluation.get("error"):
                            is_correct = evaluation["is_correct"]
                            print(f"{'✓ CORRECT' if is_correct else '✗ INCORRECT'}")
                            print(f"   Judge Reasoning: {evaluation['reasoning']}")
                            
                            # Update running accuracy counters
                            batch_evaluated_count += 1
                            if is_correct:
                                batch_correct_count += 1
                            
                            # Calculate running accuracy (overall + current batch)
                            total_evaluated_so_far = self.results["total_evaluated"] + batch_evaluated_count
                            total_correct_so_far = self.results["total_correct"] + batch_correct_count
                            running_accuracy = (total_correct_so_far / total_evaluated_so_far * 100) if total_evaluated_so_far > 0 else 0.0
                            
                            print(f"   📊 Running Accuracy: {total_correct_so_far}/{total_evaluated_so_far} correct ({running_accuracy:.1f}%)")
                        else:
                            print(f"Failed")
                    
                    # Record result
                    result = {
                        "conversation_id": conversation_id,
                        "question_type": question_type,
                        "category": category,
                        "question": question,
                        "expected_answer": str(answer_text),
                        "ai_response": ai_response,
                        "response_time": response_time,
                        "num_sessions": len(sessions),
                        "num_threads": len(thread_ids),
                        "question_thread_id": question_thread_id,
                        "evaluation": evaluation
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"\n    error on question {qa_idx}: {e}")
                    # Still record the failed question
                    result = {
                        "conversation_id": conversation_id,
                        "question_type": question_type if 'question_type' in locals() else "unknown",
                        "category": category if 'category' in locals() else -1,
                        "question": question if 'question' in locals() else "",
                        "expected_answer": str(answer_text) if 'answer_text' in locals() else "",
                        "ai_response": "",
                        "error": str(e),
                        "response_time": 0,
                        "num_sessions": len(sessions),
                        "num_threads": len(thread_ids),
                        "question_thread_id": question_thread_id
                    }
                    results.append(result)
                    # Continue to next question
                    continue
            
            # Print stats for this conversation
            self._print_conversation_stats(conversation_id, results)
            
            return results
            
        except Exception as e:
            print(f"\n   error testing questions: {e}")
            return [{
                "conversation_id": conversation_id,
                "error": str(e)
            }]
    
    async def run_test(self, dataset_items: List[Dict[str, Any]]):
        """Run the complete test on dataset items"""
        print(f"\n{'='*80}")
        print(f"LoCoMo-MC10 Dataset Test")
        print(f"{'='*80}")
        print(f"Conversations: {len(dataset_items)}")
        print(f"Verbose Logging: ON")
        print(f"Creating separate assistant for each conversation (isolated memory)")
        if DRY_RUN:
            print(f"🔍 DRY RUN MODE: Simulating execution without making actual API calls")
            print(f"{'='*80}")
        
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Test each conversation (which may have multiple questions)
            for idx, item in enumerate(dataset_items, 1):
                print(f"\n{'='*80}")
                print(f"📍 Conversation {idx}/{len(dataset_items)}")
                print(f"{'='*80}")
                
                # Create a new assistant for this conversation
                print(f"\n🆕 Creating new assistant for conversation {idx}...")
                await self.create_assistant(client)
                
                # Initialize per-conversation result tracking
                conversation_result = {
                    "conversation_id": item.get("sample_id", f"conv-{idx}"),
                    "conversation_index": idx,
                    "assistant_id": self.assistant_id,
                    "total": 0,
                    "evaluated_count": 0,
                    "correct_count": 0,
                    "accuracy_percentage": 0.0,
                    "by_type": defaultdict(lambda: {
                        "total": 0,
                        "evaluated_count": 0,
                        "correct_count": 0,
                        "accuracy_percentage": 0.0,
                        "responses": []
                    }),
                    "responses": []
                }
                
                results = await self.test_question_batch(client, item)
                
                # Update results for each question in this conversation
                for result in results:
                    question_type = result.get("question_type", "unknown")
                    evaluation = result.get("evaluation")
                    
                    # Add to questions list for tracking
                    self.results["questions"].append(result)
                    
                    # Aggregate OVERALL results
                    self.results["total"] += 1
                    self.results["by_type"][question_type]["total"] += 1
                    self.results["by_type"][question_type]["responses"].append(result)
                    self.results["all_responses"].append(result)
                    
                    # Track OVERALL evaluation results
                    if evaluation and not evaluation.get("error"):
                        is_correct = evaluation.get("is_correct", False)
                        
                        self.results["total_evaluated"] += 1
                        if is_correct:
                            self.results["total_correct"] += 1
                        
                        self.results["by_type"][question_type]["evaluated_count"] += 1
                        if is_correct:
                            self.results["by_type"][question_type]["correct_count"] += 1
                    
                    # Track PER-CONVERSATION results
                    conversation_result["total"] += 1
                    conversation_result["responses"].append(result)
                    conversation_result["by_type"][question_type]["total"] += 1
                    conversation_result["by_type"][question_type]["responses"].append(result)
                    
                    if evaluation and not evaluation.get("error"):
                        is_correct = evaluation.get("is_correct", False)
                        conversation_result["evaluated_count"] += 1
                        if is_correct:
                            conversation_result["correct_count"] += 1
                        
                        conversation_result["by_type"][question_type]["evaluated_count"] += 1
                        if is_correct:
                            conversation_result["by_type"][question_type]["correct_count"] += 1
                
                # Calculate per-conversation accuracy
                if conversation_result["evaluated_count"] > 0:
                    conversation_result["accuracy_percentage"] = (
                        conversation_result["correct_count"] / conversation_result["evaluated_count"]
                    ) * 100
                
                # Calculate per-type accuracy for this conversation
                for q_type in conversation_result["by_type"]:
                    type_data = conversation_result["by_type"][q_type]
                    if type_data["evaluated_count"] > 0:
                        type_data["accuracy_percentage"] = (
                            type_data["correct_count"] / type_data["evaluated_count"]
                        ) * 100
                
                # Store conversation result
                self.conversation_results.append(conversation_result)
                
                # Save per-conversation JSON file
                self.save_conversation_results(conversation_result, idx)
                
                # Print conversation summary
                self._print_conversation_summary(conversation_result)
                
                # Print cumulative running stats
                if self.results["total"] > 0:
                    overall_accuracy = (
                        self.results["total_correct"] / self.results["total_evaluated"] * 100
                        if self.results["total_evaluated"] > 0 else 0.0
                    )
                    print(f"\n   📊 Cumulative Overall: {self.results['total_correct']}/{self.results['total_evaluated']} correct ({overall_accuracy:.1f}%)")
            
            print(f"\nTest completed!")
    
    def save_conversation_results(self, conversation_result: Dict[str, Any], conversation_idx: int):
        """Save per-conversation results to a JSON file"""
        timestamp = int(time.time())
        conversation_id = conversation_result.get("conversation_id", f"conv-{conversation_idx}")
        filename = f"locomo_conversation_{conversation_idx}_{conversation_id}_{timestamp}.json"
        
        # Convert defaultdict to regular dict for JSON serialization
        serializable_result = {
            "conversation_id": conversation_result["conversation_id"],
            "conversation_index": conversation_result["conversation_index"],
            "assistant_id": conversation_result["assistant_id"],
            "total": conversation_result["total"],
            "evaluated_count": conversation_result["evaluated_count"],
            "correct_count": conversation_result["correct_count"],
            "accuracy_percentage": conversation_result["accuracy_percentage"],
            "by_type": {},
            "responses": conversation_result["responses"]
        }
        
        # Convert by_type defaultdict to regular dict
        for q_type, data in conversation_result["by_type"].items():
            serializable_result["by_type"][q_type] = dict(data)
        
        # Create results directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        filepath = os.path.join(results_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(serializable_result, f, indent=2)
        
        print(f"\n   💾 Saved results to: results/{filename}")
    
    def _print_conversation_summary(self, conversation_result: Dict[str, Any]):
        """Print summary for a single conversation"""
        conversation_id = conversation_result.get("conversation_id", "unknown")
        total = conversation_result.get("total", 0)
        evaluated = conversation_result.get("evaluated_count", 0)
        correct = conversation_result.get("correct_count", 0)
        accuracy = conversation_result.get("accuracy_percentage", 0.0)
        
        print(f"\n{'='*80}")
        print(f"   📊 Conversation {conversation_id} Summary")
        print(f"{'='*80}")
        print(f"   Total Questions: {total}")
        print(f"   Evaluated: {evaluated}")
        print(f"   Correct: {correct}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        # Breakdown by type
        if conversation_result.get("by_type"):
            print(f"\n   By Question Type:")
            for q_type in sorted(conversation_result["by_type"].keys()):
                type_data = conversation_result["by_type"][q_type]
                type_total = type_data.get("total", 0)
                type_evaluated = type_data.get("evaluated_count", 0)
                type_correct = type_data.get("correct_count", 0)
                type_accuracy = type_data.get("accuracy_percentage", 0.0)
                
                if type_evaluated > 0:
                    print(f"      {q_type:20s}: {type_correct:2d}/{type_evaluated:2d} ({type_accuracy:.1f}%)")
        
        print(f"{'='*80}")
    
    def _print_conversation_stats(self, conversation_id: str, results: List[Dict[str, Any]]):
        """Print statistics for a single conversation"""
        total = len(results)
        
        print(f"\n   Conversation {conversation_id} Summary:")
        print(f"      Questions: {total}")
        
        # Calculate evaluation metrics
        evaluated_results = [r for r in results if r.get("evaluation") and not r["evaluation"].get("error")]
        if evaluated_results:
            correct_count = sum(1 for r in evaluated_results if r["evaluation"].get("is_correct"))
            accuracy = (correct_count / len(evaluated_results)) * 100
            print(f"      LLM Judge: {len(evaluated_results)}/{total} evaluated | {correct_count} correct | Accuracy: {accuracy:.1f}%")
        
        # Breakdown by type
        type_stats = defaultdict(lambda: {"total": 0, "correct": 0, "evaluated": 0})
        for result in results:
            q_type = result.get("question_type", "unknown")
            type_stats[q_type]["total"] += 1
            
            evaluation = result.get("evaluation")
            if evaluation and not evaluation.get("error"):
                type_stats[q_type]["evaluated"] += 1
                if evaluation.get("is_correct"):
                    type_stats[q_type]["correct"] += 1
        
        if len(type_stats) > 1:
            print(f"      By Type:")
            for q_type, stats in sorted(type_stats.items()):
                type_total = stats["total"]
                type_line = f"         {q_type}: {type_total} questions"
                if stats["evaluated"] > 0:
                    type_acc = (stats["correct"] / stats["evaluated"]) * 100
                    type_line += f" | {stats['correct']}/{stats['evaluated']} correct | {type_acc:.1f}%"
                print(type_line)
        
        # Average response time
        response_times = [r.get("response_time", 0) for r in results if "response_time" in r]
        avg_time = sum(response_times) / len(response_times) if response_times else 0
        print(f"      Avg Response Time: {avg_time:.2f}s")
    
    def print_final_results(self):
        """Print comprehensive test results"""
        print(f"\n{'='*80}")
        print(f"Final Results")
        print(f"{'='*80}")
        
        total = self.results["total"]
        total_evaluated = self.results["total_evaluated"]
        total_correct = self.results["total_correct"]
        
        print(f"\nOverall Performance:")
        print(f"   Total Questions: {total}")
        
        # Per-Conversation Results (if available)
        if self.conversation_results:
            print(f"\n{'='*80}")
            print(f"Per-Conversation Accuracy")
            print(f"{'='*80}")
            
            conversation_accuracies = []
            for conv_result in self.conversation_results:
                conv_id = conv_result.get("conversation_id", "unknown")
                conv_idx = conv_result.get("conversation_index", 0)
                accuracy = conv_result.get("accuracy_percentage", 0.0)
                evaluated = conv_result.get("evaluated_count", 0)
                correct = conv_result.get("correct_count", 0)
                
                if evaluated > 0:
                    conversation_accuracies.append(accuracy)
                    print(f"   Conv {conv_idx:2d} ({conv_id}): {correct:2d}/{evaluated:2d} correct ({accuracy:.1f}%)")
            
            # Calculate average accuracy across conversations
            if conversation_accuracies:
                avg_conversation_accuracy = sum(conversation_accuracies) / len(conversation_accuracies)
                print(f"\n   📊 Average Accuracy Across {len(conversation_accuracies)} Conversations: {avg_conversation_accuracy:.1f}%")
        
        # LLM Judge Results
        if total_evaluated > 0:
            accuracy = (total_correct / total_evaluated) * 100
            
            print(f"\n{'='*80}")
            print(f"LLM Judge Evaluation (GPT-4o-mini)")
            print(f"{'='*80}")
            print(f"   Questions Evaluated: {total_evaluated}/{total}")
            print(f"   Correct Answers: {total_correct}/{total_evaluated}")
            print(f"   Overall Accuracy: {accuracy:.1f}%")
        else:
            print(f"\nNo LLM judge evaluations available")
        
        print(f"\nBreakdown by Question Type (Category):")
        category_names = {
            0: "single_hop (0)",
            1: "multi_hop (1)", 
            2: "temporal_reasoning (2)",
            3: "open_domain (3)",
            4: "adversarial (4)"
        }
        
        for q_type, stats in sorted(self.results["by_type"].items()):
            type_total = stats["total"]
            evaluated_count = stats.get("evaluated_count", 0)
            correct_count = stats.get("correct_count", 0)
            
            # Find category number for this type
            category_display = category_names.get(q_type, q_type) if isinstance(q_type, int) else q_type
            
            # Calculate type-specific metrics
            type_line = f"   {str(category_display):25s}: {type_total:3d} questions"
            if evaluated_count > 0:
                type_accuracy = (correct_count / evaluated_count) * 100
                type_line += f" | {correct_count}/{evaluated_count} correct | Accuracy: {type_accuracy:.1f}%"
            
            print(type_line)
        
        # Calculate average response time
        response_times = [q.get("response_time", 0) for q in self.results["questions"] if "response_time" in q]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        print(f"\nAverage Response Time: {avg_response_time:.2f}s")
        
        # Calculate average conversation accuracy
        conversation_accuracies = [
            conv.get("accuracy_percentage", 0.0) 
            for conv in self.conversation_results 
            if conv.get("evaluated_count", 0) > 0
        ]
        avg_conversation_accuracy = (
            sum(conversation_accuracies) / len(conversation_accuracies) 
            if conversation_accuracies else 0.0
        )
        
        # Save detailed results to the results directory
        results_filename = f"locomo_results_{int(time.time())}.json"
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(results_dir, results_filename)
        
        # Convert conversation_results to serializable format
        serializable_conversation_results = []
        for conv_result in self.conversation_results:
            serializable_conv = {
                "conversation_id": conv_result.get("conversation_id"),
                "conversation_index": conv_result.get("conversation_index"),
                "assistant_id": conv_result.get("assistant_id"),
                "total": conv_result.get("total"),
                "evaluated_count": conv_result.get("evaluated_count"),
                "correct_count": conv_result.get("correct_count"),
                "accuracy_percentage": conv_result.get("accuracy_percentage"),
                "by_type": {}
            }
            # Convert by_type defaultdict
            for q_type, data in conv_result.get("by_type", {}).items():
                serializable_conv["by_type"][q_type] = dict(data)
            serializable_conversation_results.append(serializable_conv)
        
        serializable_results = {
            "total_questions": total,
            "total_evaluated": total_evaluated,
            "total_correct": total_correct,
            "accuracy_percentage": float((total_correct / total_evaluated) * 100) if total_evaluated > 0 else 0.0,
            "average_conversation_accuracy": float(avg_conversation_accuracy),
            "num_conversations": len(self.conversation_results),
            "average_response_time": float(avg_response_time),
            "per_conversation_results": serializable_conversation_results,
            "by_category": {
                k: {
                    "total": v["total"],
                    "evaluated_count": v.get("evaluated_count", 0),
                    "correct_count": v.get("correct_count", 0),
                    "accuracy_percentage": float((v.get("correct_count", 0) / v.get("evaluated_count", 1)) * 100) if v.get("evaluated_count", 0) > 0 else 0.0,
                    "responses": v["responses"]
                }
                for k, v in self.results["by_type"].items()
            },
            "questions": self.results["questions"]
        }
        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
        print(f"\nDetailed results saved to: results/{results_filename}")


async def main():
    """Main entry point"""
    print("Loading LoCoMo-MC10 dataset...")
    
    dataset_items = None
    
    try:
        # Load from local JSON file (try both locations)
        dataset_items = None
        local_paths = [
            "locomo_dataset.json",  # If running from tests/ directory
            "tests/locomo_dataset.json",  # If running from project root
        ]
        
        for local_path in local_paths:
            if os.path.exists(local_path):
                print(f"Loading from local file: {local_path}")
                with open(local_path, "r") as f:
                    dataset_items = json.load(f)
                print(f"✓ Loaded {len(dataset_items)} conversations from local file")
                break
        
        if dataset_items is None:
            raise FileNotFoundError("locomo_dataset.json not found in tests/ or project root")
        
        
        # Normalize the data structure (handle both formats)
        normalized_items = []
        for item in dataset_items:
            # Check if it's the nested format with 'qa' key (which is a list of questions)
            if 'qa' in item and isinstance(item['qa'], list) and len(item['qa']) > 0:
                # Extract conversation data
                conversation_dict = item.get('conversation', {})
                session_summary = item.get('session_summary', '')
                sample_id = item.get('sample_id', 'unknown')
                
                # Extract speaker names
                speaker_a_name = conversation_dict.get('speaker_a', 'Speaker A')
                speaker_b_name = conversation_dict.get('speaker_b', 'Speaker B')
                
                # Extract all sessions from the conversation dict
                sessions = []
                session_summaries = []
                session_datetimes = []
                
                # Look for session_1, session_2, etc.
                session_num = 1
                while f'session_{session_num}' in conversation_dict:
                    session_key = f'session_{session_num}'
                    datetime_key = f'session_{session_num}_date_time'
                    
                    session_turns = conversation_dict.get(session_key, [])
                    session_datetime = conversation_dict.get(datetime_key, '')
                    
                    # Keep the full dict structure to preserve image metadata
                    formatted_turns = []
                    for turn in session_turns:
                        if isinstance(turn, dict):
                            # Keep the full dict, we'll format it during loading
                            formatted_turns.append(turn)
                        elif isinstance(turn, str):
                            # Convert string format to dict for consistency
                            # Replace speaker_a and speaker_b with actual names if it's a string
                            turn_str = turn.replace('speaker_a', speaker_a_name)
                            turn_str = turn_str.replace('speaker_b', speaker_b_name)
                            formatted_turns.append({"text": turn_str, "speaker": "Unknown"})
                    
                    if formatted_turns:
                        sessions.append(formatted_turns)
                        session_datetimes.append(session_datetime)
                        session_summaries.append('')  # No per-session summaries in this format
                    
                    session_num += 1
                
                # If no sessions found, use the entire conversation as one session
                if not sessions and isinstance(conversation_dict, list):
                    sessions = [conversation_dict]
                    session_summaries = [session_summary]
                    session_datetimes = ['']
                
                # Keep the item as a whole conversation with all QA items
                normalized_item = {
                    'sample_id': sample_id,
                    'qa': item['qa'],  # Keep all QA items together
                    'haystack_sessions': sessions,
                    'haystack_session_summaries': session_summaries,
                    'haystack_session_datetimes': session_datetimes
                }
                normalized_items.append(normalized_item)
            elif 'qa' in item and isinstance(item['qa'], dict):
                # Single QA dict format - wrap it in a list
                normalized_item = {
                    'sample_id': item.get('sample_id', 'unknown'),
                    'qa': [item['qa']],  # Wrap single QA in a list
                    'haystack_sessions': item.get('haystack_sessions', []),
                    'haystack_session_summaries': item.get('haystack_session_summaries', []),
                    'haystack_session_datetimes': item.get('haystack_session_datetimes', [])
                }
                normalized_items.append(normalized_item)
            else:
                # Already in the correct format
                normalized_items.append(item)
        
        dataset_items = normalized_items
        
        print(f"Total conversations to process: {len(dataset_items)}")
        
        # Print dataset statistics (excluding category 5)
        print(f"\nDataset Statistics (excluding category 5 - adversarial):")
        type_counts = defaultdict(int)
        total_questions = 0
        total_questions_before_filter = 0
        for item in dataset_items:
            all_qa_items = item.get('qa', [])
            total_questions_before_filter += len(all_qa_items)
            
            # Filter out category 5
            qa_items = [qa for qa in all_qa_items if qa.get("category") != 5]
            total_questions += len(qa_items)
            
            for qa in qa_items:
                category_map = {
                    1: 'single_hop',
                    2: 'temporal_reasoning',
                    3: 'multi_hop',
                    4: 'open_domain',
                    5: 'adversarial'
                }
                category = qa.get('category', 0)
                q_type = category_map.get(category, 'unknown')
                type_counts[q_type] += 1
        
        print(f"   Conversations: {len(dataset_items)}")
        print(f"   Total Questions: {total_questions} (filtered from {total_questions_before_filter})")
        print(f"\n   By Type:")
        for q_type, count in sorted(type_counts.items()):
            print(f"      {q_type:20s}: {count:4d}")
        
        # Run test
        tester = LoCoMoTester(API_BASE_URL, API_KEY)
        await tester.run_test(dataset_items)
        tester.print_final_results()
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

