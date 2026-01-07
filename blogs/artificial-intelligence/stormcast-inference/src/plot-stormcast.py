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
import sys
from typing import Callable

import cartopy
import cartopy.crs as ccrs
import cartopy.feature
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from earth2studio.data import HRRR, HRRR_FX
from matplotlib.colors import Normalize

# Fetch names and normalizations based on the variable name.
# fmt: off
NAMES: dict[str, str] = {
    'refc' : 'Composite Radar Reflectivity'
}
UNITS: dict[str,str] = {
    'refc' : 'dBZ',
}
NORMALIZATIONS: dict[str, Callable] = {
    'refc': Normalize(vmin=0, vmax=60),
}
# fmt: on


def plot_file(fname: str, max_steps: int | None = None, show: bool = False, debug: bool = False):
    if debug:
        print(f'{fname=}')
        print(f'{max_steps=}')
        print(f'{show=}')
        print(f'{debug=}')

    # Use HRRR grid definitions.
    # These definitions are directly from the earth2studio StormCast model wrapper.
    hrrr_lat_lim: tuple[int, int] = (273, 785)
    hrrr_lon_lim: tuple[int, int] = (579, 1219)
    hrrr_lat, hrrr_lon = HRRR.grid()
    model_lat = hrrr_lat[hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]]
    model_lon = hrrr_lon[hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]]

    # Use the same Lambert Conformal Conic projection as HRRR uses. This is also
    # what Earth2Studio uses internally.
    projection = ccrs.LambertConformal(
        central_longitude=262.5,
        central_latitude=38.5,
        standard_parallels=(38.5, 38.5),
        globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
    )

    # Helper for plotting the data sets.
    def plot_axis(axis, data, title):
        # Plot the field using pcolormesh.
        im = axis.pcolormesh(
            model_lon,
            model_lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap='Spectral_r',
            norm=NORMALIZATIONS[variable],
        )

        # Add coastlines and gridlines.
        axis.coastlines()
        axis.gridlines()

        # Set state lines, these will overlap with gridlines in parts, and so
        # must be drawn after them.
        axis.add_feature(
            cartopy.feature.STATES.with_scale('50m'),
            linewidth=0.5,
            edgecolor='black',
            zorder=2,
        )

        # Set title.
        axis.set_title(title)

        return im

    # Create a data source both for HRRR analysis data (more likely to correpond to the ground truth)
    # and HRRR forecast data.
    forecast_source = HRRR_FX()
    ground_truth_source = HRRR()

    # Open the model output file.
    ds = xr.open_zarr(fname, consolidated=False)
    print(f'Read in file: {fname}')

    for step, lead_time in enumerate(ds['lead_time'].to_numpy()):
        if (max_steps is not None) and (step + 1 > max_steps):
            break

        print(f'Plotting step {step}, lead time {lead_time}, max_steps {max_steps}')

        # Get the datetime and separate into date as well.
        # NOTE: Currently we only support the first forecast even if the zarr contains multiple.
        starting_time = ds['time'].to_numpy()[0]
        date = starting_time.astype('datetime64[D]')

        # Setup the plot.
        variable = 'refc'
        output_fname = f'scast-{date}-{variable}-frame-{step:02d}.jpg'

        # Use the projection in matplotlib.
        fig, axes = plt.subplots(
            nrows=1,
            ncols=3,
            subplot_kw={'projection': projection},
            figsize=(15, 6),
            layout='compressed',
        )
        axes = axes.flatten()

        # Set figure title.
        plot_time = (starting_time + lead_time).astype('datetime64[m]')
        fig.suptitle(f'{plot_time} - Lead time: {lead_time} - Quantity: {NAMES[variable]}')

        # Plot stormcast.
        # Get the target variable DataArray. For time coordinates index with 0,
        # for lead_time coordinates index with step.
        sc_data = ds[variable][0, step]
        im_sc = plot_axis(axes[0], sc_data, 'StormCast')

        # Plot HRRR forecast.
        # NOTE:
        # - earth2studios class HRRR_FX is broken, and doesn't accept non-array times even it claims it does.
        #   (It does however accept non-array lead_times for some reason.)
        # - Arguments have to be explicitly converted to python datetimes and timedeltas.
        forecast_data = forecast_source(
            time=[pd.to_datetime(starting_time).to_pydatetime()],
            lead_time=lead_time.item(),
            variable=variable,
        )
        # From forecast data we need to get that slice which corresponds to the stormcast crop.
        forecast_data_slice = forecast_data[
            0,
            0,
            0,
            hrrr_lat_lim[0] : hrrr_lat_lim[1],
            hrrr_lon_lim[0] : hrrr_lon_lim[1],
        ]
        im_forecast = plot_axis(axes[1], forecast_data_slice, 'HRRR forecast')

        # Plot HRRR analysis data.
        ground_truth_data = ground_truth_source(
            time=[pd.to_datetime(starting_time + lead_time).to_pydatetime()],
            variable=variable,
        )
        ground_truth_data_slice = ground_truth_data[
            0,
            0,
            hrrr_lat_lim[0] : hrrr_lat_lim[1],
            hrrr_lon_lim[0] : hrrr_lon_lim[1],
        ]
        im_ground_truth = plot_axis(axes[2], ground_truth_data_slice, 'HRRR analysis')

        # Add a shared colorbar, to the rightmost plot.
        cb = fig.colorbar(
            im_sc,
            ax=axes,
            orientation='vertical',
            location='right',
            # fraction=0.01,
            shrink=0.95,
        )
        cb.ax.set_title(UNITS[variable])

        print(f'Saving frame to {output_fname}')
        plt.savefig(f'{output_fname}')

        if show:
            plt.show()


def main(argv: list[str]):
    parser = argparse.ArgumentParser(description='Run StormCast.')

    parser.add_argument(
        'input',
        type=str,
        metavar='INPUT',
        help='StormCast model output',
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=None,
        help='Limit number of steps to plot.',
    )
    parser.add_argument('--show-frames', action='store_true', help='Show each frame as it is saved.')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts.')

    args = parser.parse_args(argv[1:])

    if args.debug:
        print('args', args)

    plot_file(
        fname=args.input,
        max_steps=args.max_steps,
        show=args.show_frames,
        debug=args.debug,
    )


if __name__ == '__main__':
    main(sys.argv)
