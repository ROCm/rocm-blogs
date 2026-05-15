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
import sys

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from earth2studio.data import GFS

# There are only four quantities predicted by CorrDiff, indexed by the following strings:
# - mrr: Maximum radar reflectivity [dBz]
# - t2m: Temperature at 2m height [C].
# - u10m: Wind U component at 10m height [m/s].
# - v10m: Wind V component at 10m height [m/s].

# We can plot these as three side-by-side heatmaps. The first two are already
# scalar values, and for the last panel we can show the wind speed as a scalar
# value, overlaid with the (approximate) wind vectors.

# Here we strive to use the same output projection as used in the outputs of
# the original publication, which describes the projection as the Lambert
# Conformal Conical projection "around Taiwan".
# What this mean is not fully evident, but we will take the midpoint of the
# dataset extent as the central longitude and latitude of the projection.
# The numerical values are from the publication.
lat_min, lat_max = 19.5483, 27.8446
lon_min, lon_max = 116.371, 125.568
lat_mid = 0.5 * (lat_min + lat_max)
lon_mid = 0.5 * (lon_min + lon_max)
projection = ccrs.LambertConformal(central_longitude=lon_mid, central_latitude=lat_mid)

# In addition we need the transform in which the data is defined. The
# CorrDiff outputs seem to be on lat/lon grid, so PlateCarree with default
# central longitude should be the correct choice.
# (See e.g. <https://yt-project.org/doc/visualizing/geographic_projections_and_transforms.html>)
map_transform = ccrs.PlateCarree()


def plot_lowres(args: argparse.Namespace, datetime: np.datetime64):
    # Fetch the GFS low-res data.
    data_source = GFS()
    variables = ['refc', 't2m', 'u10m', 'v10m']
    timestamp = pd.to_datetime(datetime)

    data = data_source([timestamp], variables)

    # Select only the part of the data that corresponds to the same spatial extent as the CorrDiff outputs.
    lat_ix = np.where(np.logical_and((data['lat'] >= np.floor(lat_min)), (data['lat'] <= np.ceil(lat_max))))[0]
    lon_ix = np.where(np.logical_and((data['lon'] >= np.floor(lon_min)), (data['lon'] <= np.ceil(lon_max))))[0]
    ds = data[0, :, lat_ix, lon_ix]

    # Plot the low-res data.
    figsize = (15, 6)
    fig, _ad = plt.subplot_mosaic(
        'abc',
        per_subplot_kw={
            'a': dict(projection=projection),
            'b': dict(projection=projection),
            'c': dict(projection=projection),
        },
        figsize=figsize,
        layout='compressed',
    )
    axes = np.array([_ad['a'], _ad['b'], _ad['c']])

    # Plot radar reflectivity.
    pmesh = axes[0].pcolormesh(ds['lon'], ds['lat'], ds[0], transform=map_transform, cmap=args.colormap)
    plt.colorbar(pmesh, ax=axes[0], label='Composite radar reflectivity (refc) [dBz]')

    # Plot temperature.
    pmesh = axes[1].pcolormesh(ds['lon'], ds['lat'], ds[1] - 273.15, transform=map_transform, cmap=args.colormap)
    plt.colorbar(pmesh, ax=axes[1], label='Temperature at 2m [deg C]')

    # Plot wind speed and velocity vectors.
    wind_u = ds[2].to_numpy()
    wind_v = ds[3].to_numpy()
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)

    pmesh = axes[2].pcolormesh(ds['lon'], ds['lat'], wind_speed, transform=map_transform, cmap=args.colormap)
    plt.colorbar(pmesh, ax=axes[2], label='Wind speed [m/s]')
    axes[2].quiver(ds['lon'], ds['lat'], wind_u, wind_v, transform=map_transform, width=5e-4)

    # Add coastlines and grid lines to each panel.
    for ax in axes:
        ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True, alpha=0.5)
        ax.coastlines()

    # Add a title with the date. Use pandas Timestamp for more readable string representation.
    fig.suptitle(f'GFS surface quantities for t={timestamp}')

    date = datetime.astype('datetime64[D]')
    output_fname = f'gfs-lowres-{date}.jpg'
    print(f'Saving low-res plot to {output_fname}')
    plt.savefig(output_fname)


def plot_file(args: argparse.Namespace):
    ds = xr.open_zarr(args.input, consolidated=False)

    if args.debug:
        print(ds)

    # Get an array of indexes for the sample dimension.
    sample_indexes = np.arange(len(ds['sample'].to_numpy()))

    # Check if we are plotting a specific sample, and if so, that our sample choice is within bounds.
    selected_sample = args.sample if args.sample >= 0 else None
    if selected_sample is not None:
        if selected_sample not in sample_indexes:
            raise RuntimeError(f'Selected sample index {selected_sample} not in list of valid indexes {sample_indexes}')
        sample_indexes = [selected_sample]

    # Store date and time.
    datetime = ds['time'].to_numpy()[0]
    date = datetime.astype('datetime64[D]')

    # If asked to, fetch and plot separately the low resolution input data.
    if args.plot_lowres:
        plot_lowres(args, datetime)

    # Now loop over the samples and plot each one.
    for sample_ix in sample_indexes:
        # Create the axes using the projection.
        figsize = (15, 6)
        fig, _ad = plt.subplot_mosaic(
            'abc',
            per_subplot_kw={
                'a': dict(projection=projection),
                'b': dict(projection=projection),
                'c': dict(projection=projection),
            },
            figsize=figsize,
            layout='compressed',
        )
        axes = np.array([_ad['a'], _ad['b'], _ad['c']])

        # Plot radar reflectivity.
        # NOTE: The additional indexing in ds['mrr'][0,0]  is to squeeze out the time and ensemble member axis.
        pmesh = axes[0].pcolormesh(ds['lon'], ds['lat'], ds['mrr'][0, sample_ix], transform=map_transform, cmap=args.colormap)
        plt.colorbar(pmesh, ax=axes[0], label='Radar reflectivity (mrr) [dBz]')

        # Plot temperature.
        pmesh = axes[1].pcolormesh(
            ds['lon'], ds['lat'], ds['t2m'][0, sample_ix] - 273.15, transform=map_transform, cmap=args.colormap
        )
        plt.colorbar(pmesh, ax=axes[1], label='Temperature at 2m [deg C]')

        # Plot wind speed and velocity vectors.
        # For the vectors, we need decimated data.
        lons = ds['lon'].to_numpy()
        lats = ds['lat'].to_numpy()
        wind_u = ds['u10m'][0, sample_ix].to_numpy()
        wind_v = ds['v10m'][0, sample_ix].to_numpy()
        wind_speed = np.sqrt(wind_u**2 + wind_v**2)

        df = 6
        dec_lons = lons[::df, ::df]
        dec_lats = lats[::df, ::df]
        dec_u = wind_u[::df, ::df]
        dec_v = wind_v[::df, ::df]

        pmesh = axes[2].pcolormesh(ds['lon'], ds['lat'], wind_speed, transform=map_transform, cmap=args.colormap)
        plt.colorbar(pmesh, ax=axes[2], label='Wind speed [m/s]')
        axes[2].quiver(dec_lons, dec_lats, dec_u, dec_v, transform=map_transform, width=5e-4)

        # Add coastlines and grid lines to each panel.
        for ax in axes:
            ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=True, alpha=0.5)
            ax.coastlines()

        # Add a title with the date. Use pandas Timestamp for more readable string representation.
        fig.suptitle(f'CorrDiff sample {sample_ix} for t={pd.to_datetime(datetime)}')

        # Save the plot.
        output_fname = f'corrdiff-{date}-sample-{sample_ix}.jpg'
        print(f'Saving plot to {output_fname}')
        plt.savefig(output_fname)

        # Show the plot.
        if args.show:
            plt.show()


def main(argv: list[str]):
    parser = argparse.ArgumentParser(description='Plot CorrDiff outputs.')

    parser.add_argument(
        'input',
        type=str,
        metavar='INPUT',
        help='CorrDiff output',
    )
    parser.add_argument('--sample', type=int, default=-1, help='Plot a specific sample (default: plot all samples).')
    parser.add_argument('--plot-lowres', action='store_true', help='Plot the corresponding GFS low-resolution data.')
    parser.add_argument('--show', action='store_true', help='Show the plot as it is saved.')
    parser.add_argument('--colormap', type=str, default='magma', help='Choose a matplotlib colormap (default: magma).')
    parser.add_argument('--debug', action='store_true', help='Enable debug printouts.')

    args = parser.parse_args(argv[1:])

    if args.debug:
        print('args', args)

    plot_file(args)


if __name__ == '__main__':
    main(sys.argv)
