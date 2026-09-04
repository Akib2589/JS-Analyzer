#!/usr/bin/env python3
"""
groq_js_reader_improved.py

Read JavaScript files and send them to the Groq API (chat-compatible endpoint)
for advanced security analysis.
"""

import argparse
import os
import json
import requests
import re # Added for robust JSON extraction
from pathlib import Path
from typing import List, Dict, Any

# --- Configuration ---
# Use an environment variable for API base for flexibility, default to Groq
GROQ_API_BASE = os.environ.get("GROQ_API_BASE", "https://api.groq.com/openai/v1")
CHAT_ENDPOINT = f"{GROQ_API_BASE}/chat/completions"

# Recommended context size for a high-performance Groq model (e.g., Mixtral 8x7b)
# Max tokens for payload, leaving room for the prompt and expected response
MAX_CODE_CHARS = 16000
DEFAULT_MODEL = "mixtral-8x7b-32768" # Use a known Groq model

# --- Helpers ---
def chunk_text(text: str, max_chars: int = MAX_CODE_CHARS) -> List[str]:
    """Naive chunker by characters."""
    # ... (function body is fine as is)
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
        i += max_chars
    return chunks

def extract_json_from_text(text: str) -> Dict[str, Any] | None:
    """Robustly extracts a JSON object from text, handling common issues like Markdown fences."""
    # Remove Markdown fence
    text = text.strip()
    if text.startswith('```json') and text.endswith('```'):
        text = text[7:-3].strip()
    elif text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()

    # Attempt to find the first '{...}' block (simple but often effective)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass # Fall through to full text parsing attempt

    # Attempt to parse the full cleaned text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

def build_prompt_for_chunk(filename: str, chunk_index: int, total_chunks: int, code_chunk: str) -> List[Dict]:
    """Build system+user messages for a single chunk."""
    # The system message is excellent and doesn't need changes.
    system_msg = {
        "role": "system",
        "content": (
            "You are a senior web application security engineer and bug hunter. "
            "Analyze the provided JavaScript code for security-relevant issues, interesting or "
            "valuable findings for a penetration tester (sinks, sources, dynamic eval, hard-coded endpoints, credentials, token handling, auth logic, insecure crypto, DOM sinks, risky network calls, CSP bypass vectors, etc.). "
            "For each finding provide: priority (1-5, 1 = most important), a one-line title, short rationale (why it matters), file and approximate location (line range or chunk index), and recommended safe tests to validate (explicitly DO NOT provide exploit steps or payloads). "
            "At the end, give a one-paragraph attack surface summary and list any strings that look like secrets (mask them, show only first/last 8 characters)."
        )
    }

    user_msg = {
        "role": "user",
        "content": (
            f"Filename: {filename}\nChunk: {chunk_index}/{total_chunks}\n\n"
            "=== BEGIN JAVASCRIPT ===\n"
            f"{code_chunk}\n"
            "=== END JAVASCRIPT ===\n\n"
            "Please respond in STRICT, self-contained JSON with keys: findings (array), attack_surface (string), secrets (array). DO NOT include any text, markdown, or commentary outside of the JSON object."
        )
    }

    return [system_msg, user_msg]

def call_groq_chat(messages: List[Dict], model: str) -> Dict:
    # ... (function body is fine as is)
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable not set")

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1500,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(CHAT_ENDPOINT, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

# Removed safe_mask_secret as it was unused and relies on the LLM to mask

# --- Main flow ---
def analyze_file(filepath: Path, outdir: Path, model: str):
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    # Use the consistent MAX_CODE_CHARS constant
    chunks = chunk_text(text, max_chars=MAX_CODE_CHARS)

    file_results = {
        "filename": str(filepath),
        "chunks": [],
        "combined_findings": [],
    }

    for idx, chunk in enumerate(chunks, start=1):
        messages = build_prompt_for_chunk(filepath.name, idx, len(chunks), chunk)
        print(f"[+] Sending chunk {idx}/{len(chunks)} for {filepath.name} to Groq...")

        try:
            api_resp = call_groq_chat(messages, model=model)
        except Exception as e:
            print(f"[!] API request failed for chunk {idx}: {e}")
            file_results["chunks"].append({"chunk": idx, "error": str(e)})
            continue

        # Save raw response for auditing
        # Changed raw path name for better uniqueness across files in subdirectories
        safe_filename = str(filepath).replace(os.path.sep, '_')
        raw_path = outdir / f"{safe_filename}.chunk{idx}.raw.json"
        raw_path.write_text(json.dumps(api_resp, indent=2), encoding="utf-8")

        assistant_text = ""
        try:
            # typical OpenAI-compatible structure: choices[0].message.content
            assistant_text = api_resp.get("choices", [])[0].get("message", {}).get("content", "")
        except Exception:
            assistant_text = "API response structure not as expected."

        # Use the robust JSON extractor
        parsed = extract_json_from_text(assistant_text)

        if parsed:
            # combine findings
            for f in parsed.get("findings", []):
                file_results["combined_findings"].append(f)
            file_results["chunks"].append({"chunk": idx, "parsed": parsed})
        else:
            # If assistant didn't return parsable JSON, save text
            print(f"[!] Failed to parse JSON response for chunk {idx}. Saving as text.")
            txt_path = outdir / f"{safe_filename}.chunk{idx}.assistant.txt"
            txt_path.write_text(assistant_text, encoding="utf-8")
            file_results["chunks"].append({"chunk": idx, "note": "assistant output saved as text (JSON parse failed)", "text_file": str(txt_path)})

    # Post-process combined findings: better dedupe key
    uniq = {}
    for f in file_results["combined_findings"]:
        # Dedupe key: (Title, Priority, Location)
        key = (f.get("title"), str(f.get("priority")), str(f.get("location")))
        if key not in uniq:
            uniq[key] = f
    file_results["combined_findings"] = list(uniq.values())

    # Save summary
    summary_path = outdir / f"{safe_filename}.summary.json"
    summary_path.write_text(json.dumps(file_results, indent=2), encoding="utf-8")
    print(f"[+] Saved summary to {summary_path}")

def main():
    p = argparse.ArgumentParser(description="Send JS files to Groq for security-oriented analysis.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Single JS file to analyze")
    group.add_argument("--dir", "-d", help="Directory with .js files to analyze")
    # Better default outdir name
    p.add_argument("--outdir", "-o", default=f"groq_findings_{DEFAULT_MODEL.split('-')[0]}", help="Output directory")
    p.add_argument("--model", "-m", default=DEFAULT_MODEL, help="Groq model to use (default: mixtral-8x7b-32768)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []
    if args.file:
        files = [Path(args.file)]
    elif args.dir:
        # Improved: Use rglob for deep search, ensures only .js files are targeted
        files = sorted(Path(args.dir).rglob("*.js"))

    if not files:
        print("[!] No files found to analyze.")
        return

    for f in files:
        # Skip files that are likely minified/bundler outputs unless explicitly asked
        if 'min' in f.name.lower() or 'bundle' in f.name.lower():
             print(f"[i] Skipping likely minified file: {f}")
             continue

        try:
            analyze_file(f, outdir, model=args.model)
        except Exception as e:
            print(f"[!] Failed analyzing {f}: {e}")

if __name__ == "__main__":
    main()
