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
from typing import Any, List

import earth2studio.run as run
import numpy as np
import pandas as pd
from earth2studio.data import ARCO, GFS_FX, HRRR
from earth2studio.io import ZarrBackend
from earth2studio.models.px import StormCast
from earth2studio.perturbation import Zero
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler

# Conditioning data sources currently supported by this script:
# - GFS from NOAa [default]
# - ARCO from Google.
CONDITIONING_SOURCE: dict[str, Any] = {
    'GFS': GFS_FX(),
    'ARCO': ARCO(),
}


def run_stormcast(
    starting_time: np.datetime64,
    nsteps: int,
    conditioning_data: str = 'GFS',
    ensemble: int | None = None,
    profiling: bool = False,
    debug: bool = False,
):
    # Strip starting time from the date for creating file templates.
    date = starting_time.astype('datetime64[D]')
    starting_datetime = pd.to_datetime(starting_time).to_pydatetime()
    if debug:
        print('date', date)
        print('starting_datetime', starting_datetime)

    # Create a directory for the outputs (zarr tree) if it doesn't already exist.
    os.makedirs('outputs', exist_ok=True)

    # Setup the conditioning data sources.
    cond_data_source = CONDITIONING_SOURCE[conditioning_data]

    # These two lines load the default StormCast model parameters from Nvidia's
    # NGC registry.
    # NOTE: By default, StormCast uses GFS_FX as the conditioning data source.
    package = StormCast.load_default_package()
    model = StormCast.load_model(package, conditioning_data_source=cond_data_source)

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
        # Choose the runner based on ensemble status.
        # NOTE: StormCast needs no perturbation around the mean state.
        if ensemble is not None:
            io = run.ensemble(
                time=[starting_datetime],
                nsteps=nsteps,
                nensemble=ensemble,
                prognostic=model,
                data=data,
                io=io,
                perturbation=Zero(),
            )
        else:
            io = run.deterministic(time=[starting_datetime], nsteps=nsteps, prognostic=model, data=data, io=io)

    # If we have a profiler, print out some summary statistics.
    if profiler is not None:
        print(profiler.key_averages().table(sort_by='cuda_time_total', row_limit=30))


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
    parser.add_argument('--conditioning', default='GFS', choices=list(CONDITIONING_SOURCE.keys()), help='Conditioning data source.')
    parser.add_argument('--ensemble', action='store_true', help='Use a generative ensemble.')
    parser.add_argument('--ensemble-size', type=int, default=4, help='Size of the ensemble for generative predictions.')
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts.')

    args = parser.parse_args(argv[1:])

    if args.debug:
        print('args', args)

    # Setup ensemble run if required.
    ensemble_size = None
    if args.ensemble:
        ensemble_size = args.ensemble_size

    # Run stormcast.
    run_stormcast(
        starting_time=args.datetimes,
        nsteps=args.steps,
        conditioning_data=args.conditioning,
        ensemble=ensemble_size,
        profiling=args.profile,
        debug=args.debug,
    )


if __name__ == '__main__':
    main(sys.argv)
