"""
chunking_grox_reader.py

Reads a large JavaScript file, splits it into smaller chunks to respect
API request limits, sends each chunk to the Gemini API for analysis, and then
sends all the combined chunk analyses to Gemini for a final summary report.

This version is configured for the Gemini API.

Usage:
1. Ensure you have a valid Gemini API key configured below.
2. Run: python3 chunking_grox_reader.py -f <input_js_file> -o <output_txt_file>
"""

import argparse
import os
import requests
import time
from pathlib import Path
from typing import List, Dict

# --- Configuration ---
# NOTE: The API key provided by the user is used directly here.
GEMINI_API_KEY = "AIzaSyC13QBKeHcZj7RpX2MT3wfjA03mmbbUV_U"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# The standard model for text generation tasks
DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"

# Max completion tokens for each analysis chunk
MAX_CHUNK_ANALYSIS_TOKENS = 1000
# Max completion tokens for the final summary
MAX_SUMMARY_TOKENS = 3000

# Character limit per chunk (Keeping it generous since Flash has a 1M token context window)
CHUNK_SIZE_CHARS = 10000

# --- API Caller ---

def call_gemini_api(messages: List[Dict], max_tokens: int, model: str = DEFAULT_MODEL) -> str:
    """
    Sends a chat completion request to the Gemini API.

    The 'messages' list is expected to have the system instruction as the first element
    and the user content as the second element.
    """
    if not GEMINI_API_KEY:
        return "Error: GEMINI_API_KEY is missing."

    # Construct the API URL
    API_URL = f"{GEMINI_API_BASE}/{model}:generateContent?key={GEMINI_API_KEY}"

    # Extract System Instruction and User Content from the message list
    # The structure in the calling function is messages[0]=system, messages[1]=user
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
            "temperature": 0.1,
        }
    }

    headers = {
        "Content-Type": "application/json",
    }

    # Implement basic retry/backoff for connection/transient errors
    max_retries = 3
    delay = 5 # seconds

    for attempt in range(max_retries):
        try:
            # Use a reasonable timeout
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()

            # Extract the text content from the Gemini response structure
            api_resp = resp.json()
            candidate = api_resp.get("candidates", [])[0]

            if candidate and candidate.get("content", {}).get("parts", [])[0].get("text"):
                return candidate["content"]["parts"][0]["text"]
            else:
                # Handle blocked content or other non-standard responses
                return "Error: Gemini API returned no valid content. Check safety settings or prompt."

        except requests.exceptions.HTTPError as e:
            # Handle common HTTP errors (e.g., 400 Bad Request, 429 Rate Limit)
            print(f"[!] HTTP Error on attempt {attempt + 1}: {e}")
            try:
                error_details = resp.json()
                print(f"[!] API Details: {error_details.get('error', {})}")
            except:
                pass

            if resp.status_code in [429, 500, 503] and attempt < max_retries - 1:
                print(f"[i] Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise # Re-raise for non-recoverable errors

        except requests.exceptions.ConnectionError:
            print(f"[!] Connection Error: API server not reachable at {GEMINI_API_BASE}")
            if attempt < max_retries - 1:
                print(f"[i] Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                raise

        except Exception as e:
            print(f"[!] An unexpected error occurred on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2
            else:
                raise

    return "Failed to get analysis after multiple retries."


# --- Chunking and Analysis Logic (Updated function name) ---

def chunk_file_and_analyze(filename: str, code_content: str) -> str:
    """
    Splits the code, analyzes chunks, and requests a final summary.
    Returns the final, consolidated analysis report.
    """

    # 1. Split the code into chunks
    code_length = len(code_content)

    # Simple chunking by character count
    chunks = [code_content[i:i + CHUNK_SIZE_CHARS]
              for i in range(0, code_length, CHUNK_SIZE_CHARS)]

    num_chunks = len(chunks)
    print(f"[i] File split into {num_chunks} chunks for analysis.")

    individual_analyses = []

    # 2. Analyze each chunk individually
    for i, chunk in enumerate(chunks):
        print(f"[i] Analyzing chunk {i + 1}/{num_chunks}...")

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

        # Send the system and user messages to the Gemini API
        try:
            analysis_text = call_gemini_api([system_msg, user_msg], MAX_CHUNK_ANALYSIS_TOKENS)

            # Prepend context information to the analysis before storing it
            start_char_index = i * CHUNK_SIZE_CHARS
            analysis_with_context = (
                f"--- Analysis for Part {i + 1} (Start Char Index: {start_char_index}) ---\n"
                f"{analysis_text}\n"
            )
            individual_analyses.append(analysis_with_context)

        except Exception:
            # Continue processing other chunks even if one fails
            error_analysis = f"--- Analysis for Part {i + 1} Failed ---\nError occurred during API call.\n"
            individual_analyses.append(error_analysis)
            print(f"[!] Analysis failed for chunk {i + 1}. Continuing with next chunk.")


    # 3. Request a final, consolidated summary
    print("\n[i] All chunks analyzed. Requesting final summary report...")

    combined_analysis_text = "\n".join(individual_analyses)

    # System message for final consolidation
    summary_system_msg = {
        "role": "system",
        "content": (
            "You are a senior web application security auditor. You have been provided "
            "with a set of individual analysis reports from multiple parts of a single "
            "JavaScript file. Your task is to review all the provided reports, consolidate "
            "them, remove duplicates, and generate one final, highly professional, "
            "prioritized security report. Do NOT include the individual chunk markers. "
            "Focus on high-severity issues and present the findings clearly."
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

    try:
        final_report = call_gemini_api([summary_system_msg, summary_user_msg], MAX_SUMMARY_TOKENS)
        return final_report
    except Exception:
        print("\n[!] Failed to generate final summary. Returning raw combined analysis.")
        return combined_analysis_text

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Send a large JS file to Gemini for security analysis using chunking.")
    parser.add_argument("--file", "-f", required=True, type=str, help="Path to the JavaScript file to analyze.")
    parser.add_argument("--output", "-o", required=True, type=str, help="Path to the output text file to save results.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Gemini model to use (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()

    input_path = Path(args.file)
    output_path = Path(args.output)

    # Note: We now use the global DEFAULT_MODEL for consistency, but if
    # the user supplies a different model via -m, it will be ignored unless
    # it's a valid Gemini model and we update the global variable here.
    # We will prioritize the argparse model if given.
    global DEFAULT_MODEL
    DEFAULT_MODEL = args.model # Allows overriding the model name

    if not input_path.is_file():
        print(f"[!] Error: Input file not found or is a directory: {input_path}")
        return

    # 1. Read the file
    try:
        code_content = input_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[!] Failed to read input file: {e}")
        return

    print(f"[i] Starting analysis of {input_path.name} ({len(code_content) / 1024:.2f} KB) using Gemini model: {DEFAULT_MODEL}...")

    # 2. Run the chunking and analysis process
    try:
        # Pass the desired model to the analysis function
        final_analysis = chunk_file_and_analyze(input_path.name, code_content)
    except Exception:
        print("[!] Final analysis process terminated due to API failure.")
        return

    # 3. Save the result
    try:
        output_path.write_text(final_analysis, encoding="utf-8")
        print(f"\n[+] Analysis complete! Final consolidated report saved to: {output_path.resolve()}")
    except Exception as e:
        print(f"[!] Failed to write output file: {e}")

if __name__ == "__main__":
    main()
