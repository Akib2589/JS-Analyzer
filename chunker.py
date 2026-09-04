import argparse
import os
from pathlib import Path

def create_chunks(content: str, chunk_size: int):
    """Generator that yields slices of text based on character limit."""
    for i in range(0, len(content), chunk_size):
        yield content[i : i + chunk_size]

def process_file(file_path: Path, output_base: Path, chunk_size: int):
    """Reads a file and writes its chunks into a dedicated subdirectory."""
    print(f"[i] Chunking: {file_path.name}")

    # Create a safe directory name for the chunks (e.g., source_js -> source_js_chunks)
    folder_name = f"{file_path.name.replace('.', '_')}_chunks"
    file_output_dir = output_base / folder_name
    file_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Using 'ignore' for errors to handle potential encoding issues in obfuscated JS
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[!] Failed to read {file_path.name}: {e}")
        return

    for i, chunk in enumerate(create_chunks(content, chunk_size), 1):
        chunk_filename = f"chunk_{i}.txt"
        chunk_path = file_output_dir / chunk_filename
        chunk_path.write_text(chunk, encoding="utf-8")

    print(f"[+] Created {i} chunks in: {file_output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Split text/JS files into manageable chunks for analysis.")
    parser.add_argument("--path", "-p", required=True, help="Path to a .txt/.js file or a directory.")
    parser.add_argument("--output", "-o", required=True, help="Directory to save the chunked folders.")
    parser.add_argument("--limit", "-l", type=int, default=10000, help="Character limit per chunk (default: 10000).")

    args = parser.parse_args()
    input_path = Path(args.path)
    output_base = Path(args.output)

    # Collect files
    files_to_process = []
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        # You can add more extensions here if needed
        files_to_process.extend(list(input_path.glob("*.txt")) + list(input_path.glob("*.js")))

    if not files_to_process:
        print(f"[!] No valid files found at {input_path}")
        return

    output_base.mkdir(parents=True, exist_ok=True)

    for file_path in files_to_process:
        process_file(file_path, output_base, args.limit)

if __name__ == "__main__":
    main()
