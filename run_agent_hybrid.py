import json
import click
import os
import sys

print("--- STARTING RUN_AGENT SCRIPT ---")

from agent.graph_hybrid import run_batch 

print("--- SUCCESSFUL IMPORT OF GRAPH ---")

OUTPUTS_DIR = "outputs"
os.makedirs(OUTPUTS_DIR, exist_ok=True)

@click.command()
@click.option('--batch', required=True, type=click.Path(exists=True), help='Path to the JSONL file containing questions.')
@click.option('--out', required=True, type=click.Path(), help='Path to the output JSONL file (e.g., outputs_hybrid.jsonl).')
def main(batch, out):
    # Path of the output file is relative to the current directory
    full_output_path = os.path.join(os.path.dirname(out), os.path.basename(out))
    
    # run_batch is now defined in graph_hybrid and handles file operations
    run_batch(batch, full_output_path)
    
    # Print results (optional, for CLI visibility)
    if os.path.exists(full_output_path) and os.path.getsize(full_output_path) > 0:
        print(f"\n--- Output verification from {full_output_path} ---")
        with open(full_output_path, "r", encoding="utf-8") as f:
            for line in f:
                print(line.strip())

if __name__ == "__main__":
    main()