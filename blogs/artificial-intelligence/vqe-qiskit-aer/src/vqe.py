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
import time

import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper

from utils import (
    build_ansatz,
    build_backend,
    build_optimizer,
    build_problem,
    load_config,
    print_problem_info,
    reference_energy,
    solve,
    transpile_ansatz,
)
from tracking import make_iteration_callback, log_vqe_run


def run(cfg):
    t0 = time.time()

    problem = build_problem(cfg)
    mapper = JordanWignerMapper()
    ansatz = build_ansatz(problem, cfg)
    nq = 2 * problem.num_spatial_orbitals
    backend = build_backend(cfg, nq)
    transpiled = transpile_ansatz(cfg, backend, ansatz)
    optimizer = build_optimizer(cfg)
    nuc_rep = problem.nuclear_repulsion_energy

    print_problem_info(cfg, problem, transpiled)

    ref_e = reference_energy(cfg, problem, mapper)
    if ref_e is not None:
        print(f"Reference energy:      {ref_e:.8f} Ha")
        print(f"Nuclear repulsion:     {nuc_rep:.8f} Ha")

    callback = make_iteration_callback(nuc_rep=nuc_rep)

    e0, elapsed = solve(
        problem=problem,
        mapper=mapper,
        transpiled=transpiled,
        backend=backend,
        optimizer=optimizer,
        callback=callback,
    )

    setup = time.time() - t0 - elapsed
    print(f"\nTotal energy:          {float(np.real(e0)):.8f} Ha")
    print(f"Solve time:            {elapsed:.2f}s")
    print(f"Setup time:            {setup:.2f}s")
    if ref_e is not None:
        error = abs(float(np.real(e0)) - ref_e)
        print(f"Error vs reference:    {error:.8f} Ha")

    return float(np.real(e0)), elapsed, setup, nq, transpiled.num_parameters, ref_e


def main():
    parser = argparse.ArgumentParser(description="VQE experiment runner")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--no-mlflow", action="store_true",
        help="Disable MLflow logging",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if not args.no_mlflow:
        log_vqe_run(cfg, args.config, run)
    else:
        run(cfg)


if __name__ == "__main__":
    main()
