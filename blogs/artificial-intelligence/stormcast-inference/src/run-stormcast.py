### Copyright 2025 Advanced Micro Devices, Inc.  All rights reserved.
### Licensed under the Apache License, Version 2.0 (the "License");
### you may not use this file except in compliance with the License.
### You may obtain a copy of the License at
###      http://www.apache.org/licenses/LICENSE-2.0
### Unless required by applicable law or agreed to in writing, software
### distributed under the License is distributed on an "AS IS" BASIS,
### WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
### See the License for the specific language governing permissions and
### limitations under the License.

import argparse
import os
import shutil
import sys
from contextlib import nullcontext
from typing import List

import earth2studio.run as run
import numpy as np
import pandas as pd
from earth2studio.data import HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


def run_stormcast(starting_time: np.datetime64, nsteps: int, profiling: bool = False, debug: bool = False):
    # Strip starting time from the date for creating file templates.
    date = starting_time.astype('datetime64[D]')
    starting_datetime = pd.to_datetime(starting_time).to_pydatetime()
    if debug:
        print('date', date)
        print('starting_datetime', starting_datetime)

    # Create a directory for the outputs (zarr tree) if it doesn't already exist.
    os.makedirs('outputs', exist_ok=True)

    # These two lines load the default StormCast model parameters from Nvidia's
    # NGC registry.
    package = StormCast.load_default_package()
    model = StormCast.load_model(package)

    # StormCast uses HRRR as the data source.
    data = HRRR()

    # Create a Zarr backend that persists to disk.
    # NOTE: We need to remove any pre-existing Zarr output tree since the
    # backend doesn't overwrite and just fails.
    outf = f'outputs/pred-{date}.zarr'
    if os.path.exists(outf):
        print(f'Removing existing output {outf}')
        try:
            shutil.rmtree(outf)
        except Exception as e:
            print(f'Error removing directory {outf}: {e}')
    io = ZarrBackend(outf)

    # Create a profiler if desired.
    profiler = None
    if profiling:
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler('./logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    # Run, optionally with profiling.
    with nullcontext() if profiler is None else profiler:
        io = run.deterministic([starting_datetime], nsteps, model, data, io)

    # If we have a profiler, print out some summary statistics.
    if profiler is not None:
        print(profiler.key_averages().table(sort_by='cuda_time_total', row_limit=30))

    # Optionally, one could print out the tree structure of the output.
    # print(io.root.tree())


def main(argv: List[str]):
    parser = argparse.ArgumentParser(description='Run StormCast.')

    parser.add_argument(
        'datetimes',
        type=np.datetime64,
        metavar='DATETIME',
        help='Starting dates and times (as UTC) as ISO 8601 format strings.',
    )
    parser.add_argument(
        'steps',
        metavar='STEPS',
        type=int,
        help='Number of forecasting steps (1 hour per step).',
    )
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts.')

    args = parser.parse_args(argv[1:])

    if args.debug:
        print('args', args)

    # Run stormcast.
    run_stormcast(starting_time=args.datetimes, nsteps=args.steps, profiling=args.profile, debug=args.debug)


if __name__ == '__main__':
    main(sys.argv)
