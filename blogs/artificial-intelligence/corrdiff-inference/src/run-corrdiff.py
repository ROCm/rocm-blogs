### Copyright 2026 Advanced Micro Devices, Inc.  All rights reserved.
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
from collections import OrderedDict
from contextlib import nullcontext
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from earth2studio.data import GFS, DataSource, prep_data_array
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import CorrDiffTaiwan
from earth2studio.utils.coords import map_coords, split_coords
from earth2studio.utils.type import CoordSystem
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


def create_output_file(template: str) -> Tuple[ZarrBackend, str]:
    # Create a directory for the outputs (zarr tree) if it doesn't already exist.
    os.makedirs('outputs', exist_ok=True)
    outf = os.path.join('outputs', template)

    # Create a Zarr backend that persists to disk.
    # NOTE: Pre-existing outputs with same name are truncated.
    if os.path.exists(outf):
        print(f'Removing existing output {outf}')
        try:
            shutil.rmtree(outf)
        except Exception as e:
            print(f'Error removing directory {outf}: {e}')

    return ZarrBackend(outf), outf


def fetch_input_data(
    device: torch.device, data_source: DataSource, model: CorrDiffTaiwan, starting_datetime: pd.Timestamp
) -> Tuple[torch.Tensor, CoordSystem]:
    # CorrDiff is a diagnostic model in the Earth2Studio nomenclature, which
    # means that it is primarily called using the __call__() member function.
    # For this purpose we need to manually download the data in order to feed it
    # to this function. Here 'variable' refers to the list of physical
    # quantities taken as inputs by CorrDiff.
    # See earth2studio.models.dx.corrdiff.VARIABLES for the full list.
    input_data = data_source([starting_datetime], model.input_coords()['variable'])

    # Here prep_data_array converts the inputs for PyTorch compatibility. It can
    # also optionally perform interpolation between coordinate systems.
    # The output is sent directly to the torch device.
    x, coords = prep_data_array(input_data, device=device)

    # The map_coords call interpolates the data from the output coordinates of
    # GFS to the input coordinates of CorrDiff.
    x, coords = map_coords(x, coords, model.input_coords())
    return x, coords


def setup_output(io: ZarrBackend, data_input_coords: CoordSystem, model: CorrDiffTaiwan) -> ZarrBackend:
    # Generate model output coordinate system compatible with its input coordinate system.
    output_coords = model.output_coords(model.input_coords())

    # Generate a combined coordinate system for the Zarr output.
    total_coords = OrderedDict(
        {
            'time': data_input_coords['time'],
            'sample': output_coords['sample'],
            'lat': output_coords['lat'],
            'lon': output_coords['lon'],
        }
    )

    # Set the combination of total_coords and the predicted variables as the Zarr output coordinates.
    # For the full list of variables output by CorrDiff,
    # see earth2studio.models.dx.corrdiff.OUT_VARIABLES.
    io.add_array(total_coords, output_coords['variable'])

    return io


def run_corrdiff(args: argparse.Namespace) -> None:
    # Strip starting time from the date for creating file templates.
    date = args.datetime.astype('datetime64[D]')
    starting_datetime = pd.to_datetime(args.datetime).to_pydatetime()
    if args.debug:
        print('date', date)
        print('starting_datetime', starting_datetime)

    # Create the output Zarr file.
    output_template = f'corrdiff-{date}.zarr'
    io, output_fname = create_output_file(output_template)

    # The data source for CorrDiff is the Global Forecast System (GFS) by NOAA.
    data_source = GFS()

    # These two lines load the default model parameters from Nvidia's
    # NGC registry.
    # NOTE: A pre-trained CorrDiff model is currently only available for the Taiwan region.
    package = CorrDiffTaiwan.load_default_package()
    model = CorrDiffTaiwan.load_model(package)

    # Have to explicitly move the model to the GPU if we have one.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'Running CorrDiff on device: {device}')

    # Reset peak memory use stats and prepare timers.
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        timing_start = torch.cuda.Event(enable_timing=True)
        timing_end = torch.cuda.Event(enable_timing=True)

    # Set the number of samples to generate.
    model.number_of_samples = args.samples

    # Fetch and preprocess the input data. The data will be stored in x, and the
    # coordinate system (made compatible with CorrDiff's input coordinate
    # system) is stored in coords.
    if args.debug:
        print(f'Downloading input data from: {data_source}')
    x, coords = fetch_input_data(device, data_source, model, starting_datetime)

    # Setup the Zarr storage for storing the output data.
    io = setup_output(io, coords, model)

    # Create a profiler if desired.
    profiler = None
    if args.profile:
        print('Creating profiler.')
        profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler('./logs'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )

    # Run the CorrDiff model, optionally with profiling.
    if device.type == 'cuda':
        timing_start.record()
    with nullcontext() if profiler is None else profiler:
        print(f'Running Corrdiff for t = {starting_datetime}')
        x, coords = model(x, coords)
    if device.type == 'cuda':
        timing_end.record()

    # Extract the "variables" dimension to separate tensors, and Write the all
    # of the data into the Zarr output.
    print(f'Storing output to: {output_fname}')
    io.write(*split_coords(x, coords))

    # Print out peak memory use and the timing information.
    if device.type == 'cuda':
        print(f'Peak VRAM allocated: {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB')
        print(f'Peak VRAM reserved: {torch.cuda.max_memory_reserved(device) / 1e9:.2f} GB')
        print(f'Elapsed inference time: {timing_start.elapsed_time(timing_end) / 1000:.2f} seconds')


def main(argv: List[str]):
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Run CorrDiff.')
    parser.add_argument(
        'datetime',
        type=np.datetime64,
        metavar='DATETIME',
        help='Date and time (as UTC) as an ISO 8601 format string.',
    )
    parser.add_argument('--samples', type=int, default=1, help='Number of samples to create.')
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiler.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts.')

    args = parser.parse_args(argv[1:])

    if args.debug:
        print('args', args)

    # Run CorrDiff with specified parameters.
    run_corrdiff(args)


if __name__ == '__main__':
    main(sys.argv)
