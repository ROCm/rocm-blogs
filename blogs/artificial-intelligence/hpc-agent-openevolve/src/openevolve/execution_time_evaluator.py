"""
Execution time evaluator for OpenEvolve optimization.
"""

import subprocess
import time
import re
import os


def evaluate(code: str):
    """Evaluate code execution time. Returns dict with 'error', 'fitness', 'execution_time'."""
    
    # Read file if path provided
    if os.path.isfile(code):
        with open(code, 'r') as f:
            code = f.read()
    
    # Clean code: remove LLM artifacts and extract actual code
    def clean_code(raw_code: str) -> str:
        # Extract from SEARCH/REPLACE blocks
        match = re.search(r'<<<<<<< SEARCH.*?=======(.*?)>>>>>>> REPLACE', raw_code, re.DOTALL)
        if match:
            raw_code = match.group(1).strip()
        
        # Extract from EVOLVE-BLOCK markers
        match = re.search(r'# EVOLVE-BLOCK-START(.*?)# EVOLVE-BLOCK-END', raw_code, re.DOTALL)
        if match:
            raw_code = match.group(1).strip()
        
        # Remove thinking blocks and merge markers
        lines = []
        skip = False
        for line in raw_code.split('\n'):
            lower = line.strip().lower()
            if '<think>' in lower:
                skip = True
            elif '</think>' in lower:
                skip = False
            elif skip or any(lower.startswith(m) for m in ['<<<<<<<', '=======', '>>>>>>>']):
                continue
            elif not lower.startswith('```'):
                lines.append(line)
        
        return '\n'.join(lines)
    
    cleaned_code = clean_code(code)
    
    try:
        # Warm-up run
        warmup = subprocess.run(['python'], input=cleaned_code, capture_output=True, text=True, timeout=60*5)
        if warmup.returncode != 0:
            return {'error': 0.9, 'fitness': 0.1}
        
        # Timed execution
        start = time.time()
        result = subprocess.run(['python'], input=cleaned_code, capture_output=True, text=True, timeout=60*5)
        execution_time = time.time() - start
        
        if result.returncode != 0:
            return {'error': 0.9, 'fitness': 0.1}
        
        fitness = 100.0 / (execution_time + 0.01)
        error = min(execution_time / 10.0, 1.0)
        return {'error': error, 'fitness': fitness, 'execution_time': execution_time}
        
    except (subprocess.TimeoutExpired, Exception):
        return {'error': 1.0, 'fitness': 0.0}

