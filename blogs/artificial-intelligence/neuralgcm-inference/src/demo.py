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

import os
import time
import gcsfs
import jax
import numpy as np
import pickle
import xarray
import matplotlib.pyplot as plt
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
import neuralgcm

print("=" * 60)
print("JAX Configuration:")
print("Available devices:", jax.devices())
print("Default device:", jax.devices()[0])
print("JAX version:", jax.__version__)
print("NumPy version:", np.__version__)
print("=" * 60)

# Load a pre-trained NeuralGCM model
model_name = 'v1_precip/stochastic_precip_2_8_deg.pkl'  #@param ['v1/deterministic_0_7_deg.pkl', 'v1/deterministic_1_4_deg.pkl', 'v1/deterministic_2_8_deg.pkl', 'v1/stochastic_1_4_deg.pkl', 'v1_precip/stochastic_precip_2_8_deg.pkl', 'v1_precip/stochastic_evap_2_8_deg.pkl'] {type: "string"}

gcs = gcsfs.GCSFileSystem(token='anon')
with gcs.open(f'gs://neuralgcm/models/{model_name}', 'rb') as f:
  ckpt = pickle.load(f)

model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

# Load ERA5 data from GCP/Zarr
print("Loading ERA5 data from GCP/Zarr...")
era5_path = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
full_era5 = xarray.open_zarr(
    era5_path, chunks=None, storage_options=dict(token='anon')
)

demo_start_time = '2020-03-18'
demo_end_time = '2020-03-22'
data_inner_steps = 24  # process every 24th hour

sliced_era5 = (
    full_era5
    [model.input_variables + model.forcing_variables]
    .pipe(
        xarray_utils.selective_temporal_shift,
        variables=model.forcing_variables,
        time_shift='24 hours',
    )
    .sel(time=slice(demo_start_time, demo_end_time, data_inner_steps))
    .compute()
)

print("sliced_era5:")
for key, val in sliced_era5.items():
    print(f"  {key}: {val.shape if hasattr(val, 'shape') else type(val)}")

# Regrid to NeuralGCM’s native resolution:
era5_grid = spherical_harmonic.Grid(
    latitude_nodes=full_era5.sizes['latitude'],
    longitude_nodes=full_era5.sizes['longitude'],
    latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
    longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
)
regridder = horizontal_interpolation.ConservativeRegridder(
    era5_grid, model.data_coords.horizontal, skipna=True
)
eval_era5 = xarray_utils.regrid(sliced_era5, regridder)
eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)

# initialize model state
inner_steps = 24  # save model outputs once every 24 hours
outer_steps = 4 * 24 // inner_steps  # total of 4 days
timedelta = np.timedelta64(1, 'h') * inner_steps
times = (np.arange(outer_steps) * inner_steps)  # time axis in hours

inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))
rng_key = jax.random.key(42)  # optional for deterministic models

# use persistence for forcing variables (SST and sea ice cover)
all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

print("Input shapes:")
for key, val in inputs.items():
    print(f"  {key}: {val.shape if hasattr(val, 'shape') else type(val)}")

print("input_forcings:")
for key, val in input_forcings.items():
    print(f"  {key}: {val.shape if hasattr(val, 'shape') else type(val)}")

print("all_forcings:")
for key, val in all_forcings.items():
    print(f"  {key}: {val.shape if hasattr(val, 'shape') else type(val)}")  

# make forecast
t0 = time.perf_counter()
initial_state = model.encode(inputs, input_forcings, rng_key)
jax.block_until_ready(initial_state)  # ensure JAX computation finishes
encode_time = time.perf_counter() - t0
print(f"Encode: {encode_time:.3f} s")

t0 = time.perf_counter()
final_state, predictions = model.unroll(
    initial_state,
    all_forcings,
    steps=outer_steps,
    timedelta=timedelta,
    start_with_input=True,
)
jax.block_until_ready(predictions)  # ensure JAX computation finishes
unroll_time = time.perf_counter() - t0
print(f"Unroll ({outer_steps} steps): {unroll_time:.3f} s")

t0 = time.perf_counter()
predictions_ds = model.data_to_xarray(predictions, times=times)
data_to_xarray_time = time.perf_counter() - t0
print(f"data_to_xarray: {data_to_xarray_time:.3f} s")

# Compare forecast to ERA5
# Selecting ERA5 targets from exactly the same time slice
target_trajectory = model.inputs_from_xarray(
    eval_era5
    .thin(time=(inner_steps // data_inner_steps))
    .isel(time=slice(outer_steps))
)
target_data_ds = model.data_to_xarray(target_trajectory, times=times)

combined_ds = xarray.concat([target_data_ds, predictions_ds], 'model')
combined_ds.coords['model'] = ['ERA5', 'NeuralGCM']

# Visualize ERA5 vs NeuralGCM trajectories
plot = combined_ds.specific_humidity.sel(level=850).plot(
    x='longitude', y='latitude', row='time', col='model', robust=True, aspect=2, size=2
)

# Save to PNG file
os.makedirs('./plots', exist_ok=True)
plot.fig.savefig('./plots/humidity_comparison.png', dpi=300, bbox_inches='tight')

os.makedirs('./plots', exist_ok=True)
plot = predictions_ds.precipitation_cumulative_mean.sel(surface=1, time=[24, 48, 72]).plot(
    x='longitude', y='latitude', row='time', robust=True, aspect=2, size=2
)
plot.fig.savefig('./plots/precipitation_cumulative_mean.png', dpi=300, bbox_inches='tight')

plot = predictions_ds.evaporation.sel(surface=1, time=[24, 48, 72]).plot(
    x='longitude', y='latitude', row='time', robust=True, aspect=2, size=2
)
plot.fig.savefig('./plots/evaporation.png', dpi=300, bbox_inches='tight')
