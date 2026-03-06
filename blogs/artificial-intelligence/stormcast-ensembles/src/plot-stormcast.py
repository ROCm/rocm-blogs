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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from earth2studio.data import HRRR, HRRR_FX
from matplotlib.colors import Normalize

# Fetch names and normalizations based on the variable name.
# fmt: off
NAMES: dict[str, str] = {
    'refc' : 'Composite Radar Reflectivity',
    't2m' : '2m Temperature'
}
UNITS: dict[str,str] = {
    'refc' : 'dBZ',
    't2m' : 'deg C',
}
NORMALIZATIONS: dict[str, Callable] = {
    'refc': Normalize(vmin=0, vmax=60),
    't2m': Normalize(vmin=-30, vmax=30),
}
# fmt: on

# Use HRRR grid definitions.
# The same numbers are defined in the StormCast model wrapper, and represent the StormCast grid extent.
# These are required as the StormCast grid does not cover the entire HRRR grid.
hrrr_lat_lim: tuple[int, int] = (273, 785)
hrrr_lon_lim: tuple[int, int] = (579, 1219)
hrrr_lat, hrrr_lon = HRRR.grid()
model_lat = hrrr_lat[hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]]
model_lon = hrrr_lon[hrrr_lat_lim[0] : hrrr_lat_lim[1], hrrr_lon_lim[0] : hrrr_lon_lim[1]]

# Use the same Lambert Conformal Conic projection as HRRR uses. This is also
# what Earth2Studio uses internally. It is defined in the HRRR documentation.
projection = ccrs.LambertConformal(
    central_longitude=262.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229, semiminor_axis=6371229),
)


# An implementation of the global Probability Matched Mean.
def pmm(arr: xr.DataArray):
    # Get ensemble size.
    n_ens = arr.sizes['ensemble']

    # Then compute the ensemble mean at every gridpoint.
    ens_mean = arr.mean(axis=0)
    np_mean = ens_mean.to_numpy()

    # Also create flattened index arrays into the ensemble mean array.
    ys, xs = np_mean.shape
    ymat, xmat = np.mgrid[0:ys, 0:xs]
    y_flat = ymat.flatten()
    x_flat = xmat.flatten()

    # Next, argsort the flattened ensemble mean, descending. Use these indices to sort the flattened index arrays.
    sorted_ix = np.argsort(np_mean.flatten())[::-1]
    y_sorted = y_flat[sorted_ix]
    x_sorted = x_flat[sorted_ix]

    # Pool all ensemble values across every grid point, sort descending and keep
    # every nth, where n is the number of members in the ensemble.
    ens_sorted_desc = np.sort(arr.to_numpy().flatten())[::-1]
    ens_sorted = ens_sorted_desc[::n_ens]

    # Update the ensemble mean values from the pooled and sorted ensemble values to create the PMM.
    updated_mean = ens_mean.copy()
    # NOTE: We need to do this hijink instead of directly assigning to
    # updated_mean.values[y_sorted, x_sorted], as this only seems to change a
    # copy despite what xarray's documentation claims.
    np_updated = np.zeros_like(np_mean)
    np_updated[y_sorted, x_sorted] = ens_sorted
    updated_mean.values = np_updated

    # Return the result as a DataArray.
    return updated_mean


def plot_file(
    fname: str, variable: str, max_steps: int | None = None, plot_member: int = 0, show: bool = False, debug: bool = False
):
    if debug:
        print(f'{fname=}')
        print(f'{max_steps=}')
        print(f'{plot_member=}')
        print(f'{show=}')
        print(f'{debug=}')

    # Helper for plotting the data sets.
    def plot_axis(axis, data, title):
        # Plot the field using pcolormesh.
        im = axis.pcolormesh(
            model_lon,
            model_lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap='magma',
            norm=NORMALIZATIONS[variable],
        )

        # Add coastlines and gridlines.
        axis.coastlines()
        axis.gridlines()

        # Draw state lines with 1 to 50 million scale. Some state lines overlap
        # with latlon gridlines so draw state lines on top.
        axis.add_feature(
            cartopy.feature.STATES.with_scale('50m'),
            linewidth=0.5,
            edgecolor='darkgreen',
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
    is_ensemble = False
    ds = xr.open_zarr(fname, consolidated=False)
    print(f'Read in file: {fname}')

    # Check if the file is an ensemble prediction or not.
    if 'ensemble' in ds.coords:
        print('Detected ensemble prediction.')
        is_ensemble = True
    else:
        print('Detected deterministic prediction.')

    # Store the number of time steps we have in the data.
    num_timesteps = len(ds['lead_time'].to_numpy())

    # For storing accumulated errors for error plots.
    sc_em_errors = []
    sc_pmm_errors = []
    hrrr_errors = []

    # Setup things based on whether we have an ensemble prediction or not.
    # For non-ensemble predictions we plot 1 row with 3 panels:
    # - The HRRR forecast, the HRRR ground truth and the StormCast deterministic
    #   prediction (actually just ensemble of size 1).
    # For ensemble predictions we have 3 rows with 2 panels each:
    # - HRRR forecast, HRRR ground truth
    # - One ensemble member, PMM
    # - ens. mean, running RMS error plot
    nrows = 1
    ncols = 3
    figsize = (15, 6)
    if is_ensemble:
        nrows = 3
        ncols = 2
        figsize = (10, 12)

    for step, lead_time in enumerate(ds['lead_time'].to_numpy()):
        if (max_steps is not None) and (step + 1 > max_steps):
            break

        print(f'Plotting step {step}, lead time {lead_time}, max_steps {max_steps}')

        # Get the datetime and separate into date as well.
        # NOTE: Currently we only support the first forecast even if the zarr contains multiple.
        starting_time = ds['time'].to_numpy()[0]
        date = starting_time.astype('datetime64[D]')

        # Setup the plot.
        output_fname = f'scast-{date}-{variable}-frame-{step:02d}.jpg'

        # Create the matplotlib axes using the Lambert projection.
        if is_ensemble:
            fig, _ad = plt.subplot_mosaic(
                'ab\ncd\nef',
                per_subplot_kw={
                    'a': dict(projection=projection),
                    'b': dict(projection=projection),
                    'c': dict(projection=projection),
                    'd': dict(projection=projection),
                    'e': dict(projection=projection),
                },
                figsize=figsize,
                layout='compressed',
            )
            axes = np.array([_ad['a'], _ad['b'], _ad['c'], _ad['d'], _ad['e'], _ad['f']])
        else:
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                subplot_kw={'projection': projection},
                figsize=figsize,
                layout='compressed',
            )
        axes = axes.flatten()

        # For ensembles, hide axis with index 2.
        axes[2].set_axis_off()

        # Set figure title.
        plot_time = (starting_time + lead_time).astype('datetime64[m]')
        fig.suptitle(f'{plot_time} - Lead time: {lead_time} - Quantity: {NAMES[variable]}')

        # Plot StormCast.
        # Indexing:
        # ensemble -> plot_member (if ensemble)
        # time -> 0
        # lead_time -> step
        if is_ensemble:
            # For ensemble, we plot a particular ensemble member, the PMM, the mean and the std deviation around the mean.
            data = ds[variable]
            sc_mbr_data = data[plot_member, 0, step]
            sc_avg_data = data[:, 0, step].mean(axis=0)
            sc_std_data = data[:, 0, step].std(axis=0)
            sc_pmm_data = pmm(data[:, 0, step])
            im_sc = im_sc_mbr = plot_axis(axes[2], sc_mbr_data, f'StormCast member {plot_member}')
            im_sc_pmm = plot_axis(axes[3], sc_pmm_data, 'StormCast PMM')
            im_sc_avg = plot_axis(axes[4], sc_avg_data, 'StormCast ens. mean')
            # im_sc_std = plot_axis(axes[5], sc_std_data, 'StormCast std. dev.')
        else:
            sc_data = ds[variable][0, step]
            im_sc = plot_axis(axes[2], sc_data, 'StormCast')

        # NOTE: For convenience, The HRRR plots can be individually toggled on
        # and off by changing True to False. Note that having these plots
        # enabled *will* download data from the Internet.
        if True:
            # Plot HRRR forecast.
            # NOTE:
            # - earth2studios class HRRR_FX has a bug at the time of writing,
            #   and doesn't accept non-array times even it claims it does.
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
            im_forecast = plot_axis(axes[0], forecast_data_slice, 'HRRR forecast')

        if True:
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
            im_ground_truth = plot_axis(axes[1], ground_truth_data_slice, 'HRRR analysis')

        # Finally, for ensemble plots add one panel with a running RMS error.
        # NOTE: This can be toggled off by changing True to False as well.
        if is_ensemble and True:

            def rms(ds1, ds2):
                return np.sqrt(np.sum((ds1 - ds2).to_numpy().astype(np.float64) ** 2))

            data = ds[variable]

            sc_em_errors.append(rms(sc_avg_data, ground_truth_data_slice))
            sc_pmm_errors.append(rms(sc_pmm_data, ground_truth_data_slice))
            hrrr_errors.append(rms(forecast_data_slice, ground_truth_data_slice))
            xs = np.arange(len(sc_em_errors))

            axes[5].plot(xs, sc_em_errors, label='SC ens. mean')
            axes[5].plot(xs, sc_pmm_errors, label='SC PMM')
            axes[5].plot(xs, hrrr_errors, label='HRRR')
            axes[5].set_ylabel('RMSE')
            axes[5].set_xlabel('Timestep index')
            axes[5].set_title('Error vs. ground truth')
            axes[5].set_xlim(0, num_timesteps - 1)
            axes[5].legend(loc='lower right')

        # Add a shared colorbar, position it on the right edge.
        if True:
            cb = fig.colorbar(
                # im_ground_truth,
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
    parser = argparse.ArgumentParser(description='Plot StormCast outputs.')

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
    parser.add_argument('--quantity', choices=list(NAMES.keys()), default='refc', help='Choose the quantity to plot.')
    parser.add_argument('--ensemble-member', type=int, default=0, help='Plot a specific ensemble member.')
    parser.add_argument('--show-frames', action='store_true', help='Show each frame as it is saved.')
    parser.add_argument('--colormap', type=str, default='magma', help='Choose a matplotlib colormap (default: magma).')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts.')

    args = parser.parse_args(argv[1:])

    if args.debug:
        print('args', args)

    plot_file(
        fname=args.input,
        variable=args.quantity,
        max_steps=args.max_steps,
        plot_member=args.ensemble_member,
        show=args.show_frames,
        debug=args.debug,
    )


if __name__ == '__main__':
    main(sys.argv)
