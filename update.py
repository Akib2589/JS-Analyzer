"""
chunking_grox_reader.py

Reads a large JavaScript file, splits it into smaller chunks to respect
API request limits, sends each chunk to an LLM API (via requests library) for analysis,
and then sends all the combined chunk analyses for a final summary report.

Configured for Cloudflare Workers AI (cf/meta/llama-3-8b-instruct).

Usage:
	pip install requests
	python3 chunking_grox_reader.py -f <input_js_file> -o <output_txt_file>
"""

import argparse
import os
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import requests

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Inserting the API keys/IDs you provided previously.
CF_API_ACCOUNT_ID = "3ed39b9578a3991d83de05f78ee8b2db"
CF_AUTH_TOKEN = "98meMKf0RI3Vi4wspg7BISGjuNGGzlF9Ck0dMdwB"
CF_API_BASE = f"https://api.cloudflare.com/client/v4/accounts/{CF_API_ACCOUNT_ID}/ai/run/"
CF_HEADERS = {"Authorization": f"Bearer {CF_AUTH_TOKEN}", "Content-Type": "application/json"}

DEFAULT_MODEL = "@cf/meta/llama-3-8b-instruct"

MAX_CHUNK_ANALYSIS_TOKENS = 1000
MAX_SUMMARY_TOKENS = 4000
CHUNK_SIZE_CHARS = 10000

CHUNK_RESPONSE_SCHEMA = {
	"type": "ARRAY",
	"description": "List of security findings.",
	"items": {
		"type": "OBJECT",
		"description": "A single security finding following the required structure.",
		"properties": {
			"id": {"type": "STRING"},
			"type": {"type": "STRING"},
			"severity": {"type": "STRING"},
			"confidence": {"type": "STRING"},
			"chunk_lines": {"type": "STRING"},
			"file_char_start": {"type": "INTEGER"},
			"file_char_end": {"type": "INTEGER"},
			"summary": {"type": "STRING"},
			"evidence": {"type": "STRING"},
			"exploit_concept": {"type": "STRING"},
			"recommendation": {"type": "STRING"},
			"metadata": {
				"type": "OBJECT",
				"properties": {
					"matched_pattern": {"type": "STRING"},
					"entropy_score": {"type": "NUMBER"},
					"urls": {"type": "ARRAY", "items": {"type": "STRING"}},
					"headers": {"type": "ARRAY", "items": {"type": "STRING"}},
					"libraries": {"type": "ARRAY", "items": {"type": "STRING"}},
					"notes": {"type": "STRING"}
				}
			}
		},
		"required": ["id", "type", "severity", "confidence", "chunk_lines",
					 "file_char_start", "file_char_end", "summary",
					 "evidence", "exploit_concept", "recommendation"]
	}
}


# ---------------------------------------------------------------------
# API CALLER
# ---------------------------------------------------------------------

def call_llm_api(messages: List[Dict], max_tokens: int, model: str = DEFAULT_MODEL) -> str:
	system_instruction = messages[0].get("content", "")
	user_content = messages[1].get("content", "")

	if not system_instruction or not user_content:
		return "Error: Invalid message structure for API call."

	api_url = f"{CF_API_BASE}{model}"
	payload = {
		# Cloudflare AI expects messages in this format for chat completion
		"messages": [
			{"role": "system", "content": system_instruction},
			{"role": "user", "content": user_content}
		],
		"max_tokens": max_tokens,
		"temperature": 0.1,
	}

	max_retries = 5
	delay = 35

	for attempt in range(max_retries):
		try:
			resp = requests.post(api_url, headers=CF_HEADERS, json=payload, timeout=180)
			resp.raise_for_status()
			result = resp.json()

			# Extracting the response text from the Cloudflare structure
			if 'result' in result and 'response' in result['result']:
				return result['result']['response']

			raise ValueError(f"Unexpected CF API structure: {result}")

		except requests.exceptions.RequestException as e:
			code = getattr(e.response, 'status_code', 0)
			print(f"[!] Request error on attempt {attempt+1} (Status {code}): {e}")
			if code in [429, 500, 503] and attempt < max_retries - 1:
				time.sleep(delay)
				delay *= 2
			else:
				raise
		except Exception as e:
			print(f"[!] Unexpected error on attempt {attempt+1}: {e}")
			if attempt < max_retries - 1:
				time.sleep(delay)
				delay *= 2
			else:
				raise
	return "Failed to get analysis after multiple retries."


# ---------------------------------------------------------------------
# MAIN CHUNKING + ANALYSIS
# ---------------------------------------------------------------------

def chunk_file_and_analyze(filename: str, code_content: str, model: str) -> str:
	length = len(code_content)
	chunks = [code_content[i:i + CHUNK_SIZE_CHARS] for i in range(0, length, CHUNK_SIZE_CHARS)]
	print(f"[i] File split into {len(chunks)} chunks.")

	all_findings: List[Dict[str, Any]] = []

	for i, chunk in enumerate(chunks):
		print(f"[i] Analyzing chunk {i+1}/{len(chunks)}...")
		start_char = i * CHUNK_SIZE_CHARS

		# Prompt for chunk analysis (forcing JSON output for internal use)
		system_prompt = (
			f"You are a specialized JavaScript code security analyst. "
			f"Analyze PART {i+1}/{len(chunks)} of file \"{filename}\" "
			f"(StartChar={start_char}). "
			"Identify all security vulnerabilities and risky code patterns. "
			"Output **ONLY** a JSON array strictly following the given schema. "
			"Return [] if no issues. **DO NOT** wrap the JSON in markdown fences."
		)

		user_prompt = (
			"Output schema:\n"
			f"{json.dumps(CHUNK_RESPONSE_SCHEMA, indent=2)}\n\n"
			f"=== CODE CHUNK ===\n{chunk}\n=== END ==="
		)

		analysis_text = call_llm_api(
			[{"role": "system", "content": system_prompt},
			 {"role": "user", "content": user_prompt}],
			MAX_CHUNK_ANALYSIS_TOKENS,
			model
		)

		try:
			# Robustly clean the response to ensure it's valid JSON
			cleaned = analysis_text.strip()
			start_idx = cleaned.find('[')
			end_idx = cleaned.rfind(']')

			if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
				json_str = cleaned[start_idx:end_idx + 1]
			# Fallback for LLMs that still use fences
			elif cleaned.startswith("```json") and cleaned.endswith("```"):
				json_str = cleaned[len("
