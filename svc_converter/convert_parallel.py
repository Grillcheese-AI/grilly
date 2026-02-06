#!/usr/bin/env python3
"""
Parallel SVC Converter - High-throughput version

Uses multiprocessing to maximize GPU utilization and throughput.
Splits input files and processes in parallel.

Usage:
    python convert_parallel.py --input /path/to/data --output /path/to/output --workers 4
"""

import json
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
import os
import tempfile
import shutil
from datetime import datetime

from tqdm import tqdm

# Import from main converter
from convert_to_svc import (
    SVCConverter, 
    parse_temporal_dataset,
    parse_instruct_dataset,
    parse_conversations_dataset
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    """Information about a processing chunk."""
    chunk_id: int
    input_file: Path
    output_file: Path
    start_idx: int
    end_idx: int
    parser_name: str


def count_lines(filepath: Path) -> int:
    """Count lines in a file efficiently."""
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count


def split_file_into_chunks(filepath: Path, num_chunks: int, temp_dir: Path) -> List[Path]:
    """Split a large file into smaller chunks for parallel processing."""
    
    total_lines = count_lines(filepath)
    lines_per_chunk = total_lines // num_chunks + 1
    
    chunk_files = []
    current_chunk = 0
    current_lines = 0
    current_file = None
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if current_lines == 0:
                chunk_path = temp_dir / f"chunk_{current_chunk}_{filepath.name}"
                current_file = open(chunk_path, 'w', encoding='utf-8')
                chunk_files.append(chunk_path)
            
            current_file.write(line)
            current_lines += 1
            
            if current_lines >= lines_per_chunk and current_chunk < num_chunks - 1:
                current_file.close()
                current_chunk += 1
                current_lines = 0
    
    if current_file and not current_file.closed:
        current_file.close()
    
    return chunk_files


def process_chunk(args: Tuple[Path, Path, str, int, str]) -> Tuple[int, str]:
    """
    Process a single chunk. This function runs in a separate process.
    
    Args:
        args: (input_path, output_path, parser_name, gpu_id, model_name)
    
    Returns:
        (num_processed, output_path)
    """
    input_path, output_path, parser_name, gpu_id, model_name = args
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # Initialize converter for this process
        converter = SVCConverter(model_name=model_name, gpu_id=0)  # 0 because we set CUDA_VISIBLE_DEVICES
        
        # Get parser function
        parsers = {
            'temporal': parse_temporal_dataset,
            'instruct': parse_instruct_dataset,
            'conversations': parse_conversations_dataset,
        }
        parser_func = parsers[parser_name]
        
        # Process
        count = 0
        batch = []
        batch_size = 100
        
        with open(output_path, 'w', encoding='utf-8') as out_f:
            for text, text_id, realm, metadata in parser_func(input_path):
                batch.append((text, text_id, realm, metadata))
                
                if len(batch) >= batch_size:
                    for entry in converter.process_batch(batch, batch_size=batch_size):
                        out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                        count += 1
                    batch = []
            
            # Process remaining
            if batch:
                for entry in converter.process_batch(batch, batch_size=batch_size):
                    out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    count += 1
        
        return (count, str(output_path))
    
    except Exception as e:
        logger.error(f"Error processing chunk {input_path}: {e}")
        return (0, str(output_path))


def merge_outputs(chunk_outputs: List[Path], final_output: Path):
    """Merge chunk outputs into final file."""
    with open(final_output, 'w', encoding='utf-8') as out_f:
        for chunk_file in chunk_outputs:
            if chunk_file.exists():
                with open(chunk_file, 'r', encoding='utf-8') as in_f:
                    for line in in_f:
                        out_f.write(line)


def process_file_parallel(input_path: Path, output_path: Path, parser_name: str,
                         num_workers: int, gpus: List[int], model_name: str):
    """Process a file using multiple workers."""
    
    logger.info(f"Processing {input_path} with {num_workers} workers on GPUs {gpus}")
    
    # Create temp directory for chunks
    temp_dir = Path(tempfile.mkdtemp(prefix='svc_'))
    
    try:
        # Split input file
        logger.info("Splitting input file into chunks...")
        chunk_files = split_file_into_chunks(input_path, num_workers, temp_dir)
        logger.info(f"Created {len(chunk_files)} chunks")
        
        # Prepare arguments for each worker
        chunk_outputs = []
        worker_args = []
        
        for i, chunk_file in enumerate(chunk_files):
            chunk_output = temp_dir / f"output_{i}.jsonl"
            chunk_outputs.append(chunk_output)
            gpu_id = gpus[i % len(gpus)]
            worker_args.append((chunk_file, chunk_output, parser_name, gpu_id, model_name))
        
        # Process in parallel
        logger.info(f"Starting {len(worker_args)} parallel workers...")
        
        with mp.Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_chunk, worker_args),
                total=len(worker_args),
                desc="Processing chunks"
            ))
        
        # Report results
        total_processed = sum(r[0] for r in results)
        logger.info(f"Processed {total_processed} entries across all chunks")
        
        # Merge outputs
        logger.info("Merging outputs...")
        merge_outputs(chunk_outputs, output_path)
        logger.info(f"Final output written to {output_path}")
        
    finally:
        # Cleanup temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return total_processed


def main():
    parser = argparse.ArgumentParser(description='Parallel SVC Converter')
    parser.add_argument('--input', type=str, required=True, help='Input directory')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated GPU IDs (e.g., 0,1,2,3)')
    parser.add_argument('--model', type=str, default='en_core_web_trf', help='spaCy model')
    parser.add_argument('--files', type=str, nargs='*', help='Specific files to process')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    gpus = [int(g) for g in args.gpus.split(',')]
    
    # File mappings
    file_parsers = {
        'temporal_dataset.jsonl': 'temporal',
        'temporal_dataset_events.jsonl': 'temporal',
        'instruct_anonymized_cleaned.json': 'instruct',
        'conversations_dataset_anonymized_cleaned.jsonl': 'conversations',
    }
    
    # Process files
    files_to_process = args.files if args.files else file_parsers.keys()
    
    total = 0
    for filename in files_to_process:
        filepath = input_path / filename
        if filepath.exists() and filename in file_parsers:
            out_file = output_path / f"{filepath.stem}_svc_enhanced.jsonl"
            count = process_file_parallel(
                filepath, out_file, file_parsers[filename],
                args.workers, gpus, args.model
            )
            total += count
    
    logger.info(f"Total entries processed: {total}")


if __name__ == '__main__':
    # Required for multiprocessing on some systems
    mp.set_start_method('spawn', force=True)
    main()
