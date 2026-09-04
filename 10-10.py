"""
chunking_grox_reader.py

Reads a large JavaScript file, splits it into smaller chunks to respect
API request limits, sends each chunk to the Gemini API for analysis, and then
sends all the combined chunk analyses to Gemini for a final summary report.

This version is configured for the Gemini API and implements automatic API key
rotation upon call failure to increase resilience.

Usage:
1. Ensure the GEMINI_API_KEYS list is populated with your valid keys.
2. Run: python3 chunking_grox_reader.py -f <input_js_file> -o <output_directory>
"""

import argparse
import os
import requests
import time
from pathlib import Path
from typing import List, Dict, Tuple

# --- Configuration ---
# NOTE: The user has requested to use multiple API keys for redundancy.
# The script will rotate through these keys if a call fails.
GEMINI_API_KEYS = [
    "AIzaSyC13QBKeHcZj7RpX2MT3wfjA03mmbbUV_U", # User's provided key
    "YOUR_SECOND_API_KEY_HERE",
    "YOUR_THIRD_API_KEY_HERE",
    "YOUR_FOURTH_API_KEY_HERE",
    "YOUR_FIFTH_API_KEY_HERE",
]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# The standard model for text generation tasks
DEFAULT_MODEL = "gemini-2.5-pro" # Chosen for deeper, more complex code analysis

# Max completion tokens for each analysis chunk
MAX_CHUNK_ANALYSIS_TOKENS = 1000
# Max completion tokens for the final summary
MAX_SUMMARY_TOKENS = 3000

# Character limit per chunk (Reduced to ease processing load for complex JS files)
CHUNK_SIZE_CHARS = 30000

# --- Custom Exception for API Key Failure ---

class ApiKeyFailure(Exception):
    """Custom exception raised when an API key fails likely permanently (e.g., 403 Invalid Key)."""
    pass

# --- API Caller ---

def call_gemini_api(api_key: str, messages: List[Dict], max_tokens: int, model: str = DEFAULT_MODEL) -> str:
    """
    Sends a chat completion request to the Gemini API using a single, specified key.
    Handles transient retries (429, 503) using exponential backoff.
    Raises ApiKeyFailure for likely permanent errors (400, 403).
    """
    if not api_key or "YOUR_" in api_key:
        raise ApiKeyFailure("Error: API key is missing or is a placeholder.")

    # Construct the API URL
    API_URL = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"

    # Extract System Instruction and User Content
    system_instruction = messages[0].get("content", "")
    user_content = messages[1].get("content", "")

    if not system_instruction or not user_content:
        return "Error: Invalid message structure for API call."

    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_content}]
            }
        ],
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.2,
        }
    }

    headers = {
        "Content-Type": "application/json",
    }

    max_retries = 5
    delay = 35 # Initial delay in seconds

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=180)

            # Check for permanent errors (400, 403) before re-raising.
            if resp.status_code in [400, 403]:
                print(f"[!] Critical Error ({resp.status_code}) with current key. Forcing key rotation.")
                raise ApiKeyFailure(f"API key returned status code {resp.status_code}.")

            resp.raise_for_status()

            # Extract the text content from the Gemini response structure
            api_resp = resp.json()
            candidates = api_resp.get("candidates")

            if candidates:
                candidate = candidates[0]
                content = candidate.get("content")
                parts = content.get("parts", []) if content else []

                if parts and parts[0].get("text"):
                    return parts[0]["text"]

            # Handle blocked/no-content response
            prompt_feedback = api_resp.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")

            if block_reason:
                safety_ratings = prompt_feedback.get("safetyRatings", [])
                reason_detail = f"Reason: {block_reason}. Safety Ratings: {safety_ratings}"
                return f"Error: Gemini API request blocked by filters. {reason_detail}"

            # General failure to extract text
            return "Error: Gemini API returned no valid content candidates or the response structure was unexpected."

        except requests.exceptions.HTTPError as e:
            # Handle common transient HTTP errors (429 Rate Limit, 500/503 Server Unavailable)
            print(f"[!] HTTP Error on attempt {attempt + 1}: {e}")
            try:
                error_details = resp.json()
                print(f"[!] API Details: {error_details.get('error', {})}")
            except:
                pass

            if resp.status_code in [429, 500, 503] and attempt < max_retries - 1:
                print(f"[i] Retrying with the same key in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                # Re-raise for non-recoverable errors or max retries reached on this key
                raise

        except requests.exceptions.ConnectionError as e:
            print(f"[!] Connection Error on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                print(f"[i] Retrying with the same key in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise

        except ApiKeyFailure:
            # Re-raise the custom failure to signal key rotation in the caller
            raise

        except Exception as e:
            print(f"[!] An unexpected error occurred on attempt {attempt + 1}: {e}")
            raise # Re-raise unexpected errors

    return "Failed to get analysis after multiple retries with the current key."


# --- Chunking and Analysis Logic ---

def chunk_file_and_analyze(filename: str, code_content: str, output_dir: Path, model: str) -> str:
    """
    Splits the code, analyzes chunks, and requests a final summary,
    managing API key rotation and saving intermediate results to disk.
    Returns the final, consolidated analysis report text.
    """

    # 1. Split the code into chunks
    code_length = len(code_content)
    chunks = [code_content[i:i + CHUNK_SIZE_CHARS]
              for i in range(0, code_length, CHUNK_SIZE_CHARS)]

    num_chunks = len(chunks)
    print(f"[i] File split into {num_chunks} chunks for analysis.")

    individual_analyses = []
    current_key_index = 0
    num_keys = len(GEMINI_API_KEYS)

    # 2. Analyze each chunk individually
    for i, chunk in enumerate(chunks):

        chunk_analysis_result = None
        current_chunk_key_attempts = 0

        while chunk_analysis_result is None and current_chunk_key_attempts < num_keys:
            api_key = GEMINI_API_KEYS[current_key_index]
            key_id = f"Key #{current_key_index + 1}/{num_keys}"

            print(f"\n[i] Analyzing chunk {i + 1}/{num_chunks} using {key_id}...")

            # System message for intermediate analysis
            system_msg = {
                "role": "system",
                "content": (
                    "You are a web application security engineer analyzing a *part* of a larger JavaScript file. "
                    "Analyze this specific chunk for security issues (XSS, insecure data storage, etc.). "
                    "Output ONLY a list of findings, referencing the line numbers *relative to the start of this chunk* "
                    "and include the code snippet. Be extremely concise. If no issues are found, state 'NO ISSUES FOUND'."
                )
            }

            user_msg = {
                "role": "user",
                "content": (
                    f"Analyze PART {i + 1} of {num_chunks} of file '{filename}'.\n\n"
                    "=== BEGIN CODE CHUNK ===\n"
                    f"{chunk}\n"
                    "=== END CODE CHUNK ===\n\n"
                    "Provide ONLY the concise list of findings relative to this chunk."
                )
            }

            try:
                # Attempt to call API with the current key
                analysis_text = call_gemini_api(api_key, [system_msg, user_msg], MAX_CHUNK_ANALYSIS_TOKENS, model)
                chunk_analysis_result = analysis_text # Success, exit loop

            except ApiKeyFailure as e:
                # Key rotation triggered (400/403 or placeholder key)
                print(f"[!] API Key Failure with {key_id}: {e}. Rotating key...")
                current_key_index = (current_key_index + 1) % num_keys
                current_chunk_key_attempts += 1

            except Exception as e:
                # Catch connection or other transient failures that passed through call_gemini_api
                print(f"[!] Unhandled error while processing chunk {i + 1} with {key_id}: {e}")
                # For unhandled/network errors, rotate key as a strong fallback
                print(f"[!] Rotating key to try again for chunk {i + 1}...")
                current_key_index = (current_key_index + 1) % num_keys
                current_chunk_key_attempts += 1

            # If the loop finishes without success, chunk_analysis_result will be None

        # Process result after key rotation attempts
        if chunk_analysis_result:
            # Prepend context information to the analysis
            start_char_index = i * CHUNK_SIZE_CHARS
            analysis_with_context = (
                f"--- Analysis for Part {i + 1} (Start Char Index: {start_char_index}, Key Used: {GEMINI_API_KEYS[current_key_index]}) ---\n"
                f"{chunk_analysis_result}\n"
            )
            individual_analyses.append(analysis_with_context)

            # Save individual analysis to file
            chunk_filepath = output_dir / f"chunk_{i + 1}_analysis.txt"
            chunk_filepath.write_text(chunk_analysis_result, encoding="utf-8")
            print(f"[+] Chunk {i + 1} analysis saved to {chunk_filepath.name}")

            # Advance to the next chunk, but keep the successful key for the next attempt
            current_key_index = (current_key_index + 1) % num_keys # Cycle the key after success too, for load balance

        else:
            # All keys failed for this chunk
            error_analysis = f"--- Analysis for Part {i + 1} Failed ---\nAPI call failed after attempting all {num_keys} keys.\n"
            individual_analyses.append(error_analysis)
            print(f"[!!] FATAL: Analysis failed for chunk {i + 1} after exhausting all keys. Continuing to next chunk.")


    # 3. Request a final, consolidated summary
    print("\n[i] All chunks processed. Requesting final summary report...")

    combined_analysis_text = "\n".join(individual_analyses)

    # Save the raw combined analysis for reference
    raw_path = output_dir / "combined_raw_analysis.txt"
    raw_path.write_text(combined_analysis_text, encoding="utf-8")
    print(f"[+] Raw combined analysis saved to {raw_path.name}")

    # System message for final consolidation
    summary_system_msg = {
        "role": "system",
        "content": (
            "You are a senior web application security auditor. You have been provided "
            "with a set of individual analysis reports from multiple parts of a single "
            "JavaScript file. Your task is to review all the provided reports, consolidate "
            "them, remove duplicates, and generate one final, highly professional, "
            "prioritized security report. Do NOT include the individual chunk markers (like '--- Analysis for Part X ---'). "
            "Focus on high-severity issues and present the findings clearly using Markdown headings and lists."
        )
    }

    summary_user_msg = {
        "role": "user",
        "content": (
            f"Consolidate the following analysis reports for file '{filename}' into a single, prioritized report.\n\n"
            "=== BEGIN CONSOLIDATED ANALYSES ===\n"
            f"{combined_analysis_text}\n"
            "=== END CONSOLIDATED ANALYSES ===\n"
            "Generate the final, professional security report now."
        )
    }

    # Use the current key for the final summary (or rotate if it fails)
    final_report = "Failed to generate final summary."
    summary_key_index = current_key_index
    summary_key_attempts = 0

    while summary_key_attempts < num_keys:
        api_key = GEMINI_API_KEYS[summary_key_index]
        key_id = f"Key #{summary_key_index + 1}/{num_keys}"
        print(f"[i] Attempting final summary using {key_id}...")

        try:
            final_report = call_gemini_api(api_key, [summary_system_msg, summary_user_msg], MAX_SUMMARY_TOKENS, model)
            print("[+] Final summary generated successfully.")
            break # Success, exit loop

        except ApiKeyFailure as e:
            print(f"[!] Summary Key Failure with {key_id}: {e}. Rotating key...")
            summary_key_index = (summary_key_index + 1) % num_keys
            summary_key_attempts += 1

        except Exception as e:
            print(f"[!] Unhandled error during final summary with {key_id}: {e}")
            summary_key_index = (summary_key_index + 1) % num_keys
            summary_key_attempts += 1

    return final_report

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Send a large JS file to Gemini for security analysis using chunking with key rotation.")
    parser.add_argument("--file", "-f", required=True, type=str, help="Path to the JavaScript file to analyze.")
    parser.add_argument("--output", "-o", required=True, type=str, help="Path to the output DIRECTORY to save all results.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Gemini model to use (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()

    input_path = Path(args.file)
    output_dir = Path(args.output)

    if not input_path.is_file():
        print(f"[!] Error: Input file not found or is a directory: {input_path}")
        return

    # 1. Prepare output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[i] Output directory created/verified: {output_dir.resolve()}")
    except Exception as e:
        print(f"[!] Failed to create output directory: {e}")
        return

    # 2. Read the file
    try:
        code_content = input_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[!] Failed to read input file: {e}")
        return

    print(f"[i] Starting analysis of {input_path.name} ({len(code_content) / 1024:.2f} KB) using Gemini model: {args.model}...")

    # 3. Run the chunking and analysis process
    try:
        final_analysis = chunk_file_and_analyze(input_path.name, code_content, output_dir, args.model)
    except Exception:
        print("[!] Final analysis process terminated due to unrecoverable error.")
        return

    # 4. Save the final report
    final_report_path = output_dir / "final_security_report.txt"
    try:
        final_report_path.write_text(final_analysis, encoding="utf-8")
        print(f"\n[+] Analysis complete! Final consolidated report saved to: {final_report_path.resolve()}")
    except Exception as e:
        print(f"[!] Failed to write final report file: {e}")

if __name__ == "__main__":
    main()
