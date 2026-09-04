"""
chunking_grox_reader.py

Reads a large JavaScript file or recursively processes a directory of JS files.
It splits the content into chunks, sends each chunk to the Gemini API for analysis
with robust key rotation and retries, and saves the results as individual chunk reports.

NOTE: This version removes the generation of the final, global OVERALL_SECURITY_ANALYSIS_REPORT.txt.

Usage:
1. Ensure the GEMINI_API_KEYS list is populated with your valid keys.
2. To analyze a directory: python3 chunking_grox_reader.py -p <input_directory> -o <output_directory>
3. To analyze a single file: python3 chunking_grox_reader.py -p <input_js_file> -o <output_directory>
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
    "AIzaSyClyy0qfR87M2i2A95KrpmKeX9RtsEMfwc", # Key 1
    "AIzaSyC13QBKeHcZj7RpX2MT3wfjA03mmbbUV_U", # Key 2
    "AIzaSyB-UzZT0cZek-l6Ei_PeNx9s3ohhCHD-Hc", # Key 3
    "AIzaSyB8Ga9R-tYjl89jRsbHWN3NIHS_8LKvAYg", # Key 4
    "AIzaSyA53rTbKvbfXvUv1n7KlH9knntHIh--FO4", # Key 5
]
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Reverted to gemini-2.5-pro for enhanced reasoning and deeper security analysis.
DEFAULT_MODEL = "gemini-2.5-pro"

# Max completion tokens for each analysis chunk
MAX_CHUNK_ANALYSIS_TOKENS = 10000
# Max completion tokens for the final summary (kept for compatibility with summarization logic if ever re-introduced)
MAX_SUMMARY_TOKENS = 20000

# Character limit per chunk (Reduced to ease processing load for complex JS files)
CHUNK_SIZE_CHARS = 10000

# --- Custom Exceptions for API Failures ---

class ApiKeyFailure(Exception):
    """Custom exception raised when an API key fails likely permanently (e.g., 400/403 Invalid Key)."""
    pass

class ContentExtractionFailure(Exception):
    """Custom exception raised when the API response is 200 but content cannot be extracted (e.g., empty candidates or safety block)."""
    pass

# --- API Caller ---

def call_gemini_api(api_key: str, messages: List[Dict], max_tokens: int, model: str = DEFAULT_MODEL) -> str:
    """
    Sends a chat completion request to the Gemini API using a single, specified key.
    Handles transient retries (429, 500, 503) using exponential backoff.
    Raises ApiKeyFailure for likely permanent errors (400, 403).
    Raises ContentExtractionFailure if 200 is returned but no valid content is found.
    """
    if not api_key or "YOUR_" in api_key:
        raise ApiKeyFailure("Error: API key is missing or is a placeholder.")

    # Construct the API URL
    API_URL = f"{GEMINI_API_BASE}/{model}:generateContent?key={api_key}"

    # Extract System Instruction and User Content
    # Assumes messages is structured as [System Message, User Message]
    system_instruction = messages[0].get("content", "")
    user_content = messages[1].get("content", "")

    if not system_instruction or not user_content:
        # This is a client-side preparation error, so we don't treat it as a key failure
        raise Exception("Error: Invalid message structure for API call payload.")

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

    max_retries = 5
    # Initial delay for backoff (seconds)
    delay = 35

    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=180)

            # Check for permanent errors (400, 403) before re-raising.
            if resp.status_code in [400, 403]:
                print(f"[!] Critical Error ({resp.status_code}) with current key. Forcing key rotation.")
                raise ApiKeyFailure(f"API key returned status code {resp.status_code}.")

            resp.raise_for_status()

            api_resp = resp.json()

            # --- Text Extraction (Canonical Check) ---
            candidates = api_resp.get("candidates")

            if candidates and candidates[0].get("content", {}).get("parts", []) and candidates[0]["content"]["parts"][0].get("text"):
                return candidates[0]["content"]["parts"][0]["text"]
            # ---------------------------------------------------

            # If we reach here, the response was 200 but content was missing or blocked.

            # Check for safety/content blocks if candidates is present but parts is not, or if candidates is empty.
            prompt_feedback = api_resp.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason")

            error_message = "No valid content candidates found in API response."

            if block_reason:
                safety_ratings = prompt_feedback.get("safetyRatings", [])
                reason_detail = f"Reason: {block_reason}. Safety Ratings: {safety_ratings}"
                error_message = f"API request blocked by filters. {reason_detail}"

            # Raise the custom failure to signal key rotation for this chunk
            raise ContentExtractionFailure(error_message)


        except requests.exceptions.HTTPError as e:
            # Handle common transient HTTP errors (429, 500, 503 Server Unavailable)
            print(f"[!] HTTP Error on attempt {attempt + 1}: {e}")
            try:
                error_details = resp.json()
                print(f"[!] API Details: {error_details.get('error', {})}")
            except:
                pass

            if resp.status_code in [429, 500, 503]:
                # Transient HTTP error: force immediate key rotation.
                print("[i] Transient HTTP error (429/500/503). Forcing immediate key rotation.")
                raise # Re-raise to be caught below and trigger key rotation
            else:
                # Re-raise for non-recoverable errors (e.g., 404)
                raise

        except requests.exceptions.ConnectionError as e:
            # Connection error: force immediate key rotation.
            print(f"[!] Connection Error on attempt {attempt + 1}: {e}. Forcing immediate key rotation.")
            raise

        except (ApiKeyFailure, ContentExtractionFailure):
            # Re-raise the custom failure to signal key rotation in the caller
            raise

        except Exception as e:
            print(f"[!] An unexpected error occurred on attempt {attempt + 1}: {e}")
            raise # Re-raise unexpected errors

    # If all retries for the key are exhausted, raise an unhandled exception
    raise Exception("Exhausted all retries for the current API key/chunk.")


# --- Chunking and Analysis Logic ---

def chunk_file_and_analyze(filename: str, code_content: str, output_dir: Path, model: str) -> Tuple[str, int]:
    """
    Splits the code, analyzes chunks, and manages API key rotation per chunk failure.
    Saves individual chunk analyses to the file's dedicated output directory.
    Returns the raw combined analysis and the count of chunks that failed analysis.
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
    total_skipped_chunks = 0 # Counter for chunks that failed after exhausting all keys

    # 2. Analyze each chunk individually
    for i, chunk in enumerate(chunks):

        chunk_analysis_result = None
        current_chunk_key_attempts = 0

        # Loop to retry the current chunk with the next key upon failure
        while chunk_analysis_result is None and current_chunk_key_attempts < num_keys:
            api_key = GEMINI_API_KEYS[current_key_index]
            key_id = f"Key #{current_key_index + 1}/{num_keys}"

            print(f"\n[i] Analyzing chunk {i + 1}/{num_chunks} using {key_id} (Attempt {current_chunk_key_attempts + 1})...")

            # System message for intermediate analysis (Updated to use the user's full human-readable security prompt)
            system_msg = {
                "role": "system",
                "content": (
                    "You are an expert web security researcher and bug bounty hunter. "
                    "Analyze the following JavaScript chunk as if you were conducting a professional security audit. "
                    "Your task is to find every single possible issue, vulnerability, secret, endpoint, or exploitable logic, "
                    "and explain your reasoning clearly for a human reader. "
                    "DO NOT return JSON or structured data — use human-readable text with bullet points, lists, and sections.\n\n"

                    "ANALYSIS INSTRUCTIONS\n\n"

                    "1. Secrets & Credentials\n"
                    "- Hardcoded API keys, JWTs, OAuth tokens, passwords, private keys.\n"
                    "- Base64, encoded, or obfuscated strings that may hide secrets.\n"
                    "- Any sign of API tokens or cryptographic materials.\n\n"

                    "2. Endpoints, URLs, and Hidden Paths\n"
                    "- Any internal or external URLs, /api/ endpoints, staging domains.\n"
                    "- Hidden admin panels or special routes.\n"
                    "- File paths, bucket names, or CDN links.\n\n"

                    "3. Parameters & Dynamic Fields\n"
                    "- Query or body parameters used dynamically.\n"
                    "- Keys like id, lang, redirect, product, url, token, user, etc.\n"
                    "- Any user input reflected or passed to risky sinks.\n\n"

                    "4. XSS or DOM Injection Points\n"
                    "- innerHTML, outerHTML, document.write, insertAdjacentHTML, dangerouslySetInnerHTML.\n"
                    "- Dynamic event handlers or attributes.\n"
                    "- Unescaped variables written to the DOM.\n\n"

                    "5. Dynamic Code Execution\n"
                    "- eval, new Function, setTimeout with strings, dynamic script loading.\n"
                    "- Self-modifying or obfuscated code executing runtime logic.\n\n"

                    "6. Client-side Auth & Token Handling\n"
                    "- Role or privilege checks in JS.\n"
                    "- Token storage in localStorage, sessionStorage, cookies.\n"
                    "- JWT validation client-side.\n\n"

                    "7. Network Calls & CORS\n"
                    "- fetch/axios usage with hardcoded endpoints.\n"
                    "- HTTP instead of HTTPS, 'no-cors', or credentials: include.\n\n"

                    "8. 3rd-Party Imports & Supply Chain Risks\n"
                    "- Suspicious external imports.\n"
                    "- Source maps or dev-only dependencies.\n"
                    "- Unpinned or outdated versions.\n\n"

                    "9. Privacy or Data Exfiltration\n"
                    "- Scripts sending user data or telemetry.\n"
                    "- Tracking pixels or analytics leaks.\n\n"

                    "10. Business Logic Flaws & Hidden Features\n"
                    "- Disabled or commented admin/debug code.\n"
                    "- Client-side enforced rules (price, access, limits).\n"
                    "- API calls revealing internal system logic.\n\n"

                    "11. Weak Crypto & Randomness\n"
                    "- MD5, SHA1, custom crypto, weak randoms.\n"
                    "- Secrets generated from predictable values.\n\n"

                    "12. Prototype Pollution / Object Merge Risks\n"
                    "- Object.assign, merge, extend on user-controlled input.\n\n"

                    "13. Obfuscation or Encoded Logic\n"
                    "- atob/btoa usage, encoded strings, self-decoding loops.\n"
                    "- Hidden URLs or function names revealed after decoding.\n\n"

                    "OUTPUT FORMAT\n\n"
                    "Write your output in clear, human-readable text, using this layout:\n\n"
                    "[Category Name]\n"
                    "- Finding summary\n"
                    "- Why it’s important\n"
                    "- Code snippet or variable reference\n"
                    "- Possible impact or exploit scenario\n"
                    "- Suggested mitigation\n\n"
                    "At the end, summarize the highest-risk findings with reasoning on what to investigate further.\n\n"
                    "Be exhaustive — even small hints or clues matter. Never skip potential findings."
                )
            }

            user_msg = {
                "role": "user",
                "content": (
                    f"Analyze PART {i + 1} of {num_chunks} of file '{filename}'.\n\n"
                    "=== BEGIN CODE CHUNK ===\n"
                    f"{chunk}\n"
                    "=== END CODE CHUNK ===\n\n"
                    "Follow the instructions in the system message strictly. Only include findings relevant to client-side JavaScript security and clearly explain each finding for a human reader. "
                    "Avoid unrelated style or performance notes."
                )
            }

            try:
                # Attempt to call API with the current key
                analysis_text = call_gemini_api(api_key, [system_msg, user_msg], MAX_CHUNK_ANALYSIS_TOKENS, model)
                chunk_analysis_result = analysis_text # Success, exit loop

            except (ApiKeyFailure, ContentExtractionFailure, Exception) as e:
                # Catch failures (400/403, Content Missing/Blocked, Transient Errors)
                error_type = type(e).__name__

                # --- NEW LOGIC: Skip chunk immediately on ContentExtractionFailure ---
                if error_type == 'ContentExtractionFailure':
                    print(f"[!] Content Extraction Error with {key_id}: {e}. Skipping chunk {i + 1} immediately.")
                    # Set attempts to max to terminate the while loop immediately
                    current_chunk_key_attempts = num_keys
                elif error_type == 'ApiKeyFailure':
                    print(f"[!] Key Failure (400/403) with {key_id}. Forcing rotation...")
                    current_key_index = (current_key_index + 1) % num_keys
                    current_chunk_key_attempts += 1
                else:
                    # ConnectionError, HTTPError (transient errors), or unexpected Exception
                    print(f"[!] Transient Error ({error_type}) with {key_id}: {e}. Retrying chunk with next key...")
                    current_key_index = (current_key_index + 1) % num_keys
                    current_chunk_key_attempts += 1

                if current_chunk_key_attempts == num_keys:
                    # Break the while loop if all keys have been exhausted or ContentExtractionFailure occurred.
                    break

        # Process result after key rotation attempts
        if chunk_analysis_result:
            # Prepend context information to the analysis
            start_char_index = i * CHUNK_SIZE_CHARS

            # Determine the key used for the successful attempt
            successful_key_index = (current_key_index - current_chunk_key_attempts + num_keys) % num_keys
            if current_chunk_key_attempts == 0:
                 successful_key_index = current_key_index

            successful_key = GEMINI_API_KEYS[successful_key_index]

            analysis_with_context = (
                f"--- Analysis for Part {i + 1} (Start Char Index: {start_char_index}, Key Used: {successful_key}) ---\n"
                f"{chunk_analysis_result}\n"
            )
            individual_analyses.append(analysis_with_context)

            # Save individual analysis to file
            chunk_filepath = output_dir / f"chunk_{i + 1}_analysis.txt"
            chunk_filepath.write_text(chunk_analysis_result, encoding="utf-8")
            print(f"[+] Chunk {i + 1} analysis saved to {chunk_filepath.name}")

        else:
            # This block is executed if ContentExtractionFailure occurred or if all keys failed for other reasons.
            total_skipped_chunks += 1
            final_error = str(e) if 'e' in locals() else 'Analysis failed after exhausting retries or immediate content block.'
            keys_attempted_msg = f"API call failed after attempting {current_chunk_key_attempts} key(s)."

            error_analysis = (
                f"--- Analysis for Part {i + 1} Failed ---\n"
                f"{keys_attempted_msg} Final error: {final_error}\n"
            )
            individual_analyses.append(error_analysis)
            print(f"[!!] FATAL: Analysis failed for chunk {i + 1}. Continuing to next chunk.")


    # Combine all analyses into one block for the main report (still useful as a per-file summary)
    combined_analysis_text = "\n".join(individual_analyses)

    # Save the raw combined analysis for reference within the file's output folder
    raw_path = output_dir / "combined_raw_analysis.txt"
    raw_path.write_text(combined_analysis_text, encoding="utf-8")
    print(f"[+] Raw combined analysis (per-file summary) saved to {raw_path.name}")

    return combined_analysis_text, total_skipped_chunks

# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser(description="Send large JS files to Gemini for security analysis using chunking with key rotation.")
    parser.add_argument("--path", "-p", required=True, type=str, help="Path to the JavaScript file or DIRECTORY to analyze.")
    parser.add_argument("--output", "-o", required=True, type=str, help="Path to the base output DIRECTORY to save all results.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Gemini model to use (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()

    input_path = Path(args.path)
    output_dir = Path(args.output)

    # 1. Prepare output base directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[i] Base output directory created/verified: {output_dir.resolve()}")
    except Exception as e:
        print(f"[!] Failed to create base output directory: {e}")
        return

    # 2. Determine files to analyze
    files_to_analyze = []
    if input_path.is_file() and input_path.suffix.lower() == '.js':
        files_to_analyze.append(input_path)
    elif input_path.is_dir():
        # Find all .js files recursively
        files_to_analyze.extend(input_path.glob('**/*.js'))
    else:
        print(f"[!] Error: Input path must be a .js file or a directory containing .js files: {input_path}")
        return

    if not files_to_analyze:
        print(f"[i] No .js files found in the specified path: {input_path}")
        return

    print(f"[i] Found {len(files_to_analyze)} .js file(s) for analysis.")

    # 3. Process each file
    for file_path in files_to_analyze:

        # Calculate relative path for reporting and output folder naming
        try:
            # If input_path is a dir, this gives 'sub/file.js'
            relative_path_str = str(file_path.relative_to(input_path))
        except ValueError:
            # If input_path is a file, this gives 'file.js'
            relative_path_str = file_path.name

        print(f"\n==========================================================================")
        print(f"STARTING ANALYSIS FOR: {relative_path_str}")
        print(f"==========================================================================")

        # Create a unique output subdirectory for the file results
        # Replace path separators and dots (except for the last one) with underscores for safe folder naming
        folder_name = relative_path_str.replace(os.path.sep, '__').replace('.', '_')
        file_output_dir = output_dir / folder_name

        try:
            file_output_dir.mkdir(parents=True, exist_ok=True)
            print(f"[i] File-specific output directory: {file_output_dir.resolve()}")
        except Exception as e:
            error_msg = f"Failed to create file output directory {file_output_dir}: {e}"
            print(f"[!] {error_msg}. Skipping file.")
            continue

        # Read the file
        try:
            code_content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            error_msg = f"Failed to read input file {file_path}: {e}"
            print(f"[!] {error_msg}")
            continue

        # Run the chunking and analysis process
        try:
            raw_analysis, skipped_count = chunk_file_and_analyze(file_path.name, code_content, file_output_dir, args.model)

            # Print status summary
            if skipped_count > 0:
                print(f"[!] WARNING: {skipped_count} chunk(s) skipped for {relative_path_str} due to API failure after exhausting retries or content block.")
            else:
                print(f"[+] All chunks processed successfully for {relative_path_str}.")

        except Exception as e:
            error_msg = f"Analysis process for {relative_path_str} terminated due to unrecoverable error: {e}"
            print(f"[!] {error_msg}")

    # 4. Save the final overall report (Removed)
    print(f"\n[+] Processing complete. Individual chunk reports for each file are saved in subdirectories under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
