# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import pandas as pd
import subprocess
import tempfile

from amdsmi import (
    amdsmi_init,
    amdsmi_get_processor_handles,
    amdsmi_get_rocm_version,
    amdsmi_get_gpu_asic_info,
    amdsmi_get_gpu_driver_info,
    AmdSmiException,
)
from fastmcp import FastMCP
from pathlib import Path

mcp = FastMCP("rocm-profile-server")


@mcp.tool()
def get_rocm_info():
    """
    Retrieve ROCm installation information.
    Includes:
      - ROCm version
      - ROCm installation path

    Returns:
        dict: A dictionary containing the ROCm information.
    """
    try:
        rocm_load_status, version_message = amdsmi_get_rocm_version()
        return {
            "success": True,
            "info": {
                "load_status": rocm_load_status,
                "version_message": version_message,
            },
        }
    except AmdSmiException as e:
        return {"success": False, "error": e}


@mcp.tool()
def get_gpu_driver_info():
    """
    Retrieve information for the GPU driver.
    Includes:
      - driver name
      - driver version
      - driver build date

    Returns:
        dict: A dictionary containing the GPU driver information.
    """
    try:
        amdsmi_init()
        devices = amdsmi_get_processor_handles()
        driver_info = amdsmi_get_gpu_driver_info(devices[0])
        return {"success": True, "info": driver_info}
    except AmdSmiException as e:
        return {"success": False, "error": e}


@mcp.tool()
def get_gpu_asic_info():
    """
    Retrieve ASIC information for the given GPU.
    Includes:
       - marketing name
       - vendor ID
       - vendor name
       - device ID
       - revision ID
       - ASIC serial
       - OAM ID
       - number of compute units on ASIC
       - hardware graphics version

    Returns:
        dict: A dictionary containing the ASIC information.
    """
    try:
        amdsmi_init()
        devices = amdsmi_get_processor_handles()
        asic_info = amdsmi_get_gpu_asic_info(devices[0])
        return {"success": True, "info": asic_info}
    except AmdSmiException as e:
        return {"success": False, "error": e}


@mcp.tool()
def run_rocprofv3_kernel_summary(
    run_name: str, target_executable_path: str, target_parameters: str = None
) -> dict:
    """
    Runs rocprofv3 to profile a target executable and returns a summary of kernel activity.

    Resulting summary table is returned in summary field of the returned dictionary.
    All results data are saved in CSV format in the output directory.

    Args:
        target_executable_path: Path to the target executable to profile
        target_parameters: Parameters to pass to the target executable (optional)

    Returns:
        Dictionary containing profiling results (success, summary, output_dir).
    """
    try:
        # Create directory for results
        temp_dir = tempfile.mkdtemp(prefix="rocprofv3_kernel_summary_")
        results_dir = os.path.join(temp_dir, run_name)

        # Build rocprofv3 command
        target_executable_path = os.path.abspath(target_executable_path)

        cmd = ["rocprofv3", "--kernel-trace", "--stats", "-S", "-T"]
        cmd.extend(["--output-directory", results_dir])
        cmd.extend(["--output-file", run_name])
        cmd.extend(["--output-format", "csv"])
        cmd.extend(["--"])
        cmd.extend([target_executable_path])
        cmd.extend(target_parameters.split(" "))

        # Run profiling
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Read stats file if it exists
        stats_csv = results_dir + "/" + run_name + "_kernel_stats.csv"
        stats = None
        if os.path.exists(stats_csv):
            with open(stats_csv, "r") as f:
                stats = f.read()

        # Return results
        if result.returncode == 0:
            return {
                "success": True,
                "summary": stats,
                "output_dir": results_dir,
            }
        else:
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],  # Limit stdout size
                "stderr": result.stderr[-1000:],  # Limit stderr size
            }

    # Handle exceptions
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout after 300 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def run_rocprofv3_occupancy_summary(
    run_name: str, target_executable_path: str, target_parameters: str = None
) -> dict:
    """
    Runs rocprofv3 to profile a target executable and returns occupancy percentage for kernels.

    Resulting summary table is returned in summary field of the returned dictionary.
    All results data are saved in CSV format in the output directory.

    Args:
        target_executable_path: Path to the target executable to profile
        target_parameters: Parameters to pass to the target executable (optional)

    Returns:
        Dictionary containing profiling results (success, summary, output_dir).
    """
    temp_dir = None

    try:
        # Create directory for results
        temp_dir = tempfile.mkdtemp(prefix="rocprofv3_occupancy_")
        results_dir = os.path.join(temp_dir, run_name)

        # Build rocprofv3 command
        target_executable_path = os.path.abspath(target_executable_path)

        cmd = ["rocprofv3", "--pmc", "OccupancyPercent", "-T"]
        cmd.extend(["--output-directory", results_dir])
        cmd.extend(["--output-file", run_name])
        cmd.extend(["--output-format", "csv"])
        cmd.extend(["--"])
        cmd.extend([target_executable_path])
        cmd.extend(target_parameters.split(" "))

        # Run profiling
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=temp_dir
        )

        # Read counters file if it exists and compute summary
        counters_csv = results_dir + "/" + run_name + "_counter_collection.csv"
        stats = None
        if os.path.exists(counters_csv):
            df = pd.read_csv(counters_csv)
            df = df.rename(columns={"Counter_Value": "Occupancy_Percentage"})
            stats = (
                df.groupby("Kernel_Name")
                .agg(
                    {
                        "Occupancy_Percentage": ["mean", "min", "max"],
                        "VGPR_Count": "mean",
                        "SGPR_Count": "mean",
                        "Workgroup_Size": "mean",
                    }
                )
                .to_string()
            )

        # Return results
        if result.returncode == 0:
            return {
                "success": True,
                "summary": stats,
                "output_dir": results_dir,
            }
        else:
            return {
                "success": False,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],  # Limit stdout size
                "stderr": result.stderr[-1000:],  # Limit stderr size
            }

    # Handle exceptions
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout after 300 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def run_rocprof_compute_profile(
    run_name: str, target_executable_path: str, target_parameters: str = None
) -> dict:
    """
    Runs rocprof-compute to profile a target executable and saves the profiling data
    to an output directory. This data can be later used for analysis.

    Does not perform roofline analysis.

    The output path is /path/to/tempdir/run_name/, for example:
    /tmp/rocprof_compute_xyz/my_run_name/.

    Args:
        run_name: Name for the profiling run
        target_executable_path: Path to the target executable to profile
        target_parameters: Parameters to pass to the target executable (optional)

    Returns:
        Dictionary containing profiling results (success, returncode, stdout, stderr, output_dir).
    """
    try:
        # Create directory for results
        temp_dir = tempfile.mkdtemp(prefix="rocprof_compute_")
        results_dir = os.path.join(temp_dir, run_name)

        # Build rocprof-compute command
        target_executable_path = os.path.abspath(target_executable_path)

        cmd = ["rocprof-compute", "profile"]
        cmd.extend(["--name", run_name])
        cmd.extend(["--path", results_dir])
        cmd.extend(["--no-roof"])
        cmd.extend(["--", target_executable_path, target_parameters])

        # Run profiling
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=temp_dir
        )

        # Return results
        if result.returncode == 0:
            return {
                "success": True,
                "output_dir": results_dir,
            }
        else:
            return {
                "success": False,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],  # Limit stdout size
                "stderr": result.stderr[-1000:],  # Limit stderr size
            }

    # Handle exceptions
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout after 300 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def run_rocprof_compute_analyze(
    target_dir: str, block_id: str = None, kernel_id: str = None
) -> dict:
    """
    Runs rocprof-compute to analyze profiling data in a given directory.

    The target path should be the complete path /path/to/tempdir/run_name/, for example:
    /tmp/rocprof_compute_xyz/my_run_name/.

    Args:
        target_dir: Path to the directory containing profiling data
        block_id: Specify a subset of metrics to analyze (optional)
            Options include:
                0 -> Top Stats
                1 -> System Info
                2 -> System Speed-of-Light
                3 -> Memory Chart
                4 -> Roofline
        kernel_id: Specify a kernel to analyze (optional)
    """
    try:
        # Build rocprof-compute command
        cmd = ["rocprof-compute", "analyze"]
        cmd.extend(["--path", target_dir])
        if block_id is not None:
            cmd.extend(["--block", block_id])
        if kernel_id is not None:
            cmd.extend(["--kernel", kernel_id])

        # Run profiling
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=target_dir
        )

        # Return results
        if result.returncode == 0:
            return {
                "success": True,
                "stdout": result.stdout,
            }
        else:
            return {
                "success": False,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],  # Limit stdout size
                "stderr": result.stderr[-1000:],  # Limit stderr size
            }

    # Handle exceptions
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout after 300 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@mcp.tool()
def run_roofline_analysis(
    run_name: str,
    target_executable_path: str,
    target_parameters: str = None,
    target_kernel: str = None,
) -> dict:
    """
    Runs rocprof-compute to perform roofline analysis on a target executable.

    Main result is roofline figure saved in PDF format.
    Other result include the profiling data used to generate the roofline chart.
    All data is saved in output directory.

    Args:
        target_executable_path: Path to the target executable to profile
        target_parameters: Parameters to pass to the target executable (optional)
        target_kernel: Specific kernel to profile (optional)

    Returns:
        Dictionary containing profiling results (success, roofline_pdf_path, output_dir).
    """
    try:
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix="rocprof_roofline_")

        # Build rocprof-compute command
        target_executable_path = os.path.abspath(target_executable_path)

        cmd = ["rocprof-compute", "profile"]
        cmd.extend(["--name", run_name])
        cmd.extend(["--path", temp_dir])
        cmd.extend(["--roof-only"])
        cmd.extend(["--device", "0"])
        if target_kernel is not None:
            cmd.extend(["--kernel", target_kernel])
        cmd.extend(["--", target_executable_path, target_parameters])

        # Run profiling
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300, cwd=temp_dir
        )

        # Find roofline graph path
        results_path = Path(temp_dir)
        matches = list(results_path.glob("empirRoof_*.pdf"))
        roofline_pdf_path = matches[0] if matches else None

        # Return results
        if result.returncode == 0:
            return {
                "success": True,
                "roofline_pdf_path": str(roofline_pdf_path),
                "output_dir": temp_dir,
            }
        else:
            return {
                "success": False,
                "returncode": result.returncode,
                "stdout": result.stdout[-1000:],  # Limit stdout size
                "stderr": result.stderr[-1000:],  # Limit stderr size
            }

    # Handle exceptions
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Timeout after 300 seconds"}
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    mcp.run()
