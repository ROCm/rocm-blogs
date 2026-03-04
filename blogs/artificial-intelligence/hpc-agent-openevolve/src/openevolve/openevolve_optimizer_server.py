import os
import asyncio
import logging
import json
import tempfile
import sys
from openevolve.config import load_config
from fastmcp import FastMCP

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mcp = FastMCP('openevolve-optimizer-server')


async def optimize_code(
    script_content: str,
    config_file: str,
    evaluation_file: str,
    max_iterations: int = 2
) -> dict:
    """
    Generic optimization function that runs OpenEvolve with specified config and evaluator.
    
    Args:
        script_content: Python script with code to optimize (can be code string or file path)
        config_file: Path to YAML configuration file
        evaluation_file: Path to evaluation file
        max_iterations: Maximum optimization iterations (default: 2)
    
    Returns:
        Dictionary with 'original_code', 'optimized_code', 'best_score', and 'metrics'
    """
    initial_program_path = None
    temp_dir = None
    
    try:
        # Create temporary directory for OpenEvolve output
        temp_dir = tempfile.mkdtemp(prefix='openevolve_')
        
        # Handle both file paths and code strings for initial program
        if os.path.isfile(script_content):
            logger.info(f"Using existing file: {script_content}")
            initial_program_path = script_content
            with open(script_content, 'r') as f:
                original_code = f.read()
        else:
            logger.info(f"Creating temporary file for code string: {len(script_content)} bytes")
            # Create temporary file for initial program
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir=temp_dir) as f:
                initial_program_path = f.name
                f.write(script_content)
                original_code = script_content
        
        logger.info(f"Starting optimization: {len(original_code)} bytes, config={config_file}")
        logger.info(f"Initial program: {initial_program_path}")
        logger.info(f"Evaluation file: {evaluation_file}")
        
        # Load configuration from YAML using OpenEvolve's load_config
        config = load_config(config_file)
        
        # Create a temporary script to run OpenEvolve in a subprocess
        script_content_for_subprocess = f"""
import os
import sys
import json
import asyncio
from openevolve import OpenEvolve
from openevolve.config import load_config

async def run_openevolve():
    initial_program_path = {repr(initial_program_path)}
    evaluation_file = {repr(evaluation_file)}
    config_file = {repr(config_file)}
    max_iterations = {max_iterations}
    temp_dir = {repr(temp_dir)}
    
    # Load configuration
    config = load_config(config_file)
    
    # Create OpenEvolve instance
    evolve = OpenEvolve(
        initial_program_path=initial_program_path,
        evaluation_file=evaluation_file,
        config=config,
        output_dir=temp_dir
    )
    
    # Run evolution
    result = await evolve.run(iterations=max_iterations)
    
    if result is None:
        print(json.dumps({{"success": False, "error": "No result returned"}}), file=sys.stderr)
        sys.exit(1)
    
    # Extract fitness info
    fitness_info = {{}}
    if hasattr(result, 'metrics') and result.metrics:
        fitness_info['metrics'] = result.metrics
        if isinstance(result.metrics, dict):
            fitness_info['best_score'] = result.metrics.get('fitness') or result.metrics.get('score')
    
    # Output result as JSON
    output = {{
        "success": True,
        "optimized_code": result.code,
        "best_score": fitness_info.get('best_score'),
        "metrics": fitness_info.get('metrics', {{}})
    }}
    print(json.dumps(output))
    sys.stdout.flush()

if __name__ == "__main__":
    asyncio.run(run_openevolve())
"""
        
        # Create temporary script file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_path = f.name
            f.write(script_content_for_subprocess)
        
        try:
            # Run OpenEvolve in subprocess
            logger.info(f"Starting evolution with {max_iterations} iterations in subprocess")
            logger.info("This may take a while - LLM calls will be logged below...")
            
            # Calculate timeout (estimate based on iterations and population size)
            timeout = max_iterations * config.database.population_size * config.llm.timeout
            if timeout < 600:
                timeout = 600  # Minimum 10 minutes
            
            proc = await asyncio.create_subprocess_exec(
                sys.executable, script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                logger.error(f"Subprocess timed out after {timeout}s")
                proc.kill()
                await proc.wait()
                raise Exception(f"Optimization timed out after {timeout/60:.1f} minutes")
            
            if stderr:
                logger.debug(f"Subprocess stderr:\n{stderr.decode()}")
            
            if proc.returncode != 0:
                error_msg = stderr.decode()[:500] if stderr else "Unknown error"
                raise Exception(f"Subprocess failed with return code {proc.returncode}: {error_msg}")
            
            # Parse result from subprocess
            result_data = json.loads(stdout.decode())
            
            if not result_data.get('success', False):
                raise Exception(f"Optimization failed: {result_data.get('error', 'Unknown error')}")
            
            logger.info("Evolution completed successfully")
            
            return {
                "success": True,
                "original_code": original_code,
                "optimized_code": result_data['optimized_code'],
                "best_score": result_data.get('best_score'),
                "metrics": result_data.get('metrics', {})
            }
            
        finally:
            # Clean up temporary script
            if os.path.exists(script_path):
                try:
                    os.unlink(script_path)
                except Exception:
                    pass
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
    finally:
        # Note: We don't delete temp_dir or initial_program_path if it was a user-provided file
        # Only delete if we created it as a temporary file
        if initial_program_path and temp_dir and initial_program_path.startswith(temp_dir):
            try:
                os.unlink(initial_program_path)
            except Exception:
                pass


@mcp.tool()
async def optimize_code_execution_time(
    script_content: str,
    max_iterations: int = 10
) -> dict:
    """
    Optimize code execution time using OpenEvolve evolutionary optimization.
    
    Args:
        script_content: Python script with code to optimize
        max_iterations: Maximum optimization iterations (default: 10)
    
    Returns:
        Dictionary with 'original_code', 'optimized_code', 'best_score', and 'metrics'
    """
    # Use absolute paths for config and evaluator files
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(base_dir, 'execution_time_config.yaml')
    evaluation_file = os.path.join(base_dir, 'execution_time_evaluator.py')
    
    return await optimize_code(
        script_content=script_content,
        config_file=config_file,
        evaluation_file=evaluation_file,
        max_iterations=max_iterations
    )


if __name__ == "__main__":
    mcp.run()

