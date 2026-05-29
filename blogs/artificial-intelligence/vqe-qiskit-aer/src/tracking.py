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

import os
import time

import mlflow
import numpy as np

MLFLOW_EXPERIMENT = "VQE"


def _flat_params(d, prefix=""):
    """Flatten a nested dict into dotted keys for ``mlflow.log_params``."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flat_params(v, key))
        else:
            out[key] = v
    return out


def _run_name(cfg):
    geom = cfg.get("molecule", {}).get("geometry", "unknown")
    atoms = {s.split()[0] for s in geom.split(";") if s.strip()}
    mol_tag = "".join(sorted(atoms)).lower() or "unknown"
    sim = cfg.get("simulator", {})
    ndev = int(sim.get("num_devices", 1) or 1)
    gpu_tag = f"_{ndev}gpu" if ndev > 1 else ""
    return f"{mol_tag}_{cfg.get('basis', '?')}_{cfg.get('ansatz', '?')}_{cfg.get('optimizer', '?')}{gpu_tag}"


def make_iteration_callback(nuc_rep=0.0):
    """Per-iteration table + MLflow step metrics.

    ``nuc_rep`` is added to the electronic energy so both the table and the
    MLflow ``energy`` metric show **total** energy in Hartree.
    """
    step = 0
    t0 = t_prev = None
    e_prev = None

    def callback(nfev, _parameters, energy, _metadata):
        nonlocal step, t0, t_prev, e_prev
        t = time.monotonic()
        if step == 0:
            t0 = t_prev = t
            print(
                f"| {'nfev':>6} | {'E_total [Ha]':>14} | {'dE [Ha]':>14} | "
                f"{'dt [s]':>8} | {'elapsed [s]':>9} |"
            )
        e_total = float(np.real(energy)) + nuc_rep
        de = 0.0 if e_prev is None else e_total - e_prev
        de_str = "\u2014" if e_prev is None else f"{de:+.6e}"
        e_prev = e_total
        dt = 0.0 if step == 0 else t - t_prev
        dt_str = "\u2014" if step == 0 else f"{dt:.4f}"
        t_prev = t
        elapsed = t - t0
        step += 1
        print(
            f"| {nfev:>6} | {e_total:>14.8f} | {de_str:>14} | "
            f"{dt_str:>8} | {elapsed:>9.4f} |"
        )
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "energy": float(e_total),
                    "dt_s": float(dt),
                    "elapsed_s": float(elapsed),
                },
                step=nfev,
            )

    return callback


def log_vqe_run(cfg, config_path, run_fn):
    """Set up MLflow tracking, execute ``run_fn(cfg)``, and log params/metrics."""
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
    )
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=_run_name(cfg)):
        mlflow.log_params(_flat_params(cfg))
        mlflow.set_tag("config_file", config_path)

        e0, elapsed, setup, nq, npar, ref_e = run_fn(cfg)

        mlflow.log_params({
            "num_qubits": int(nq),
            "num_params": int(npar),
        })
        metrics = {
            "total_energy_Ha": float(e0),
            "solve_seconds": float(elapsed),
            "setup_seconds": float(setup),
        }
        if ref_e is not None:
            metrics["reference_energy_Ha"] = float(ref_e)
            metrics["error_vs_exact_Ha"] = float(abs(e0 - ref_e))
        mlflow.log_metrics(metrics)
