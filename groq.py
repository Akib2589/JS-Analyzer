#!/usr/bin/env python3
"""
groq_js_reader.py

Read JavaScript files and send them to the Groq API (Responses/chat-compatible endpoint)
asking Groq to act as an advanced bug-hunter and return prioritized, safe findings.

Safety note: the prompt explicitly prohibits exploit instructions. Do NOT send
unredacted secrets to third-party APIs unless permitted by the engagement owner.
"""

import argparse
import os
import json
import requests
from pathlib import Path
from typing import List, Dict

GROQ_API_BASE = "https://api.groq.com/openai/v1"  # Groq exposes OpenAI-compatible endpoints
CHAT_ENDPOINT = f"{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}/chat/completions"  # chat completions endpoint
# If your Groq account/docs show a different endpoint (Responses API), adapt accordingly.

# --- Helpers ---
def chunk_text(text: str, max_chars: int = 20000) -> List[str]:
    """Naive chunker by characters. Adjust max_chars if you have token limits."""
    chunks = []
    i = 0
    L = len(text)
    while i < L:
        chunk = text[i:i + max_chars]
        chunks.append(chunk)
        i += max_chars
    return chunks

def build_prompt_for_chunk(filename: str, chunk_index: int, total_chunks: int, code_chunk: str) -> List[Dict]:
    """
    Build system+user messages for a single chunk. We use a short system instruction
    and then the JS code labeled with filename and chunk index.
    """
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
            "Please respond in JSON with keys: findings (array), attack_surface (string), secrets (array)."
        )
    }

    return [system_msg, user_msg]

def call_groq_chat(messages: List[Dict], model: str = "gpt-oss-20b") -> Dict:
    """
    Send a chat completion request to Groq's OpenAI-compatible chat endpoint.
    Returns parsed JSON response.
    - model default is an example; replace with a model available to your account.
    """
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

def safe_mask_secret(s: str) -> str:
    if len(s) <= 16:
        return s[:2] + "..." + s[-2:]
    return s[:8] + "..." + s[-8:]

# --- Main flow ---
def analyze_file(filepath: Path, outdir: Path, model: str):
    text = filepath.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_text(text, max_chars=18000)
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
        raw_path = outdir / f"{filepath.name.replace('/', '_')}.chunk{idx}.raw.json"
        raw_path.write_text(json.dumps(api_resp, indent=2), encoding="utf-8")

        # Attempt to parse assistant message content as JSON (user asked for JSON)
        assistant_text = ""
        try:
            # typical OpenAI-compatible structure: choices[0].message.content
            assistant_text = api_resp.get("choices", [])[0].get("message", {}).get("content", "")
        except Exception:
            assistant_text = str(api_resp)

        parsed = None
        try:
            parsed = json.loads(assistant_text)
            # combine findings
            for f in parsed.get("findings", []):
                file_results["combined_findings"].append(f)
        except Exception:
            # If assistant didn't return strict JSON, save text and attempt to extract
            txt_path = outdir / f"{filepath.name}.chunk{idx}.assistant.txt"
            txt_path.write_text(assistant_text, encoding="utf-8")
            file_results["chunks"].append({"chunk": idx, "note": "assistant output saved as text", "text_file": str(txt_path)})
            continue

        file_results["chunks"].append({"chunk": idx, "parsed": parsed})

    # Post-process combined findings: simple dedupe by title+line
    uniq = {}
    for f in file_results["combined_findings"]:
        key = (f.get("title"), str(f.get("location")))
        if key not in uniq:
            uniq[key] = f
    file_results["combined_findings"] = list(uniq.values())

    # Save summary
    summary_path = outdir / f"{filepath.name}.summary.json"
    summary_path.write_text(json.dumps(file_results, indent=2), encoding="utf-8")
    print(f"[+] Saved summary to {summary_path}")
    return file_results

def main():
    p = argparse.ArgumentParser(description="Send JS files to Groq for security-oriented analysis.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", help="Single JS file to analyze")
    group.add_argument("--dir", "-d", help="Directory with .js files to analyze")
    p.add_argument("--outdir", "-o", default="groq_findings", help="Output directory")
    p.add_argument("--model", "-m", default="gpt-oss-20b", help="Groq model to use (change as needed)")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = []
    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(Path(args.dir).glob("**/*.js"))

    if not files:
        print("[!] No files found to analyze.")
        return

    for f in files:
        try:
            analyze_file(f, outdir, model=args.model)
        except Exception as e:
            print(f"[!] Failed analyzing {f}: {e}")

if __name__ == "__main__":
    main()
