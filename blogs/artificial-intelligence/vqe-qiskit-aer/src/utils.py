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

import time
import numpy as np
import yaml
from scipy.sparse.linalg import eigsh
from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit
from qiskit_algorithms.optimizers import COBYLA, SPSA, SLSQP
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_algorithms import VQE
from qiskit_nature.second_q.algorithms import GroundStateEigensolver

_DEFAULTS = {
    "molecule": {"geometry": "Li 0 0 0; H 0 0 1.6"},
    "basis": "sto-3g",
    "charge": 0,
    "spin": 0,
    "ansatz": "uccsd",
    "optimizer": "spsa",
    "maxiter": 100,
    "simulator": {
        "method": "statevector",
        "device": "GPU",
        "precision": "single",
        "num_devices": 1,
        "blocking_qubits": None,
    },
    "transpile": {"optimization_level": 3},
    "reference": "auto",
}

_EFFICIENT_SU2_DEFAULTS = {"reps": 2, "entanglement": "linear"}


def _deep_merge(base, override):
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(_DEFAULTS, user)
    if cfg.get("ansatz", "").lower() == "efficient_su2":
        cfg["efficient_su2"] = _deep_merge(
            _EFFICIENT_SU2_DEFAULTS, cfg.get("efficient_su2")
        )
    return cfg


def build_problem(cfg):
    driver = PySCFDriver(
        atom=cfg["molecule"]["geometry"],
        basis=cfg["basis"],
        charge=int(cfg["charge"]),
        spin=int(cfg["spin"]),
        unit=DistanceUnit.ANGSTROM,
    )
    return driver.run()


def build_ansatz(problem, cfg):
    mapper = JordanWignerMapper()
    nso = problem.num_spatial_orbitals
    hf = HartreeFock(nso, problem.num_particles, mapper)
    name = cfg["ansatz"].lower()
    if name == "uccsd":
        return UCCSD(nso, problem.num_particles, mapper, initial_state=hf)
    if name == "efficient_su2":
        nq = 2 * nso
        es = cfg.get("efficient_su2", {})
        return EfficientSU2(
            num_qubits=nq,
            reps=int(es.get("reps", 2)),
            entanglement=es.get("entanglement", "linear"),
            initial_state=hf,
        )
    raise ValueError(f"Unknown ansatz: {cfg['ansatz']}")


def build_backend(cfg, num_qubits):
    sim = cfg.get("simulator", {})
    kwargs = {
        "method": sim.get("method", "statevector"),
        "device": sim.get("device", "GPU"),
        "precision": sim.get("precision", "single"),
    }
    num_devices = int(sim.get("num_devices", 1) or 1)
    if num_devices > 1 and kwargs["device"].upper() != "CPU":
        bq = sim.get("blocking_qubits")
        if bq is None:
            bq = num_qubits - num_devices.bit_length()
        kwargs["blocking_enable"] = True
        kwargs["blocking_qubits"] = int(bq)
        n_chunks = 2 ** (num_qubits - int(bq))
        print(
            f"Multi-GPU: num_devices={num_devices} "
            f"blocking_qubits={bq} chunks={n_chunks}"
        )
    return AerSimulator(**kwargs)


def transpile_ansatz(cfg, backend, ansatz):
    opt_level = int((cfg.get("transpile") or {}).get("optimization_level", 3))
    pm = generate_preset_pass_manager(optimization_level=opt_level, backend=backend)
    return pm.run(ansatz)


_OPTIMIZERS = {"cobyla": COBYLA, "spsa": SPSA, "slsqp": SLSQP}


def build_optimizer(cfg):
    name = cfg["optimizer"].lower()
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {cfg['optimizer']}")
    return _OPTIMIZERS[name](maxiter=int(cfg["maxiter"]))


def solve(problem, mapper, transpiled, backend, optimizer, callback=None):
    estimator = AerEstimator.from_backend(backend)
    vqe = VQE(estimator, transpiled, optimizer)
    vqe.initial_point = np.zeros(transpiled.num_parameters)
    vqe.callback = callback
    calc = GroundStateEigensolver(mapper, vqe)
    t0 = time.time()
    result = calc.solve(problem)
    elapsed = time.time() - t0
    return float(np.real(result.total_energies[0])), elapsed


def classical_exact_energy(problem, mapper):
    """Exact ground-state energy via sparse diagonalisation (< 2^20 dim)."""
    qubit_op = mapper.map(problem.hamiltonian.second_q_op())
    nq = qubit_op.num_qubits
    if nq > 20:
        return None
    nuc_rep = problem.nuclear_repulsion_energy
    H_sparse = qubit_op.to_matrix(sparse=True)
    eigenvalues, _ = eigsh(H_sparse, k=1, which="SA")
    return float(np.real(eigenvalues[0])) + nuc_rep


def classical_ccsdt_energy(cfg):
    """CCSD(T) total energy via PySCF."""
    from pyscf import gto, scf, cc

    mol = gto.M(
        atom=cfg["molecule"]["geometry"].replace(";", "\n"),
        basis=cfg["basis"],
        charge=int(cfg["charge"]),
        spin=int(cfg["spin"]),
        unit="Angstrom",
    )
    mf = scf.RHF(mol).run(verbose=0)
    mycc = cc.CCSD(mf).run(verbose=0)
    et = mycc.ccsd_t()
    return float(mycc.e_tot + et)


def reference_energy(cfg, problem, mapper):
    """Classical reference energy: auto | exact | ccsdt | none."""
    mode = cfg.get("reference", "auto").lower()
    if mode == "none":
        return None
    if mode == "exact":
        return classical_exact_energy(problem, mapper)
    if mode == "ccsdt":
        return classical_ccsdt_energy(cfg)
    exact = classical_exact_energy(problem, mapper)
    if exact is not None:
        return exact
    return classical_ccsdt_energy(cfg)


def _format_bytes(n_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024 or unit == "TB":
            return f"{int(n_bytes)} B" if unit == "B" else f"{n_bytes:.2f} {unit}"
        n_bytes /= 1024.0


def print_problem_info(cfg, problem, transpiled):
    nq = transpiled.num_qubits
    prec = str(cfg.get("simulator", {}).get("precision", "double")).lower()
    bpc = 8 if prec in ("single", "float", "float32", "fp32") else 16

    print("=" * 50)
    print("Config")
    print("=" * 50)
    print(yaml.dump(cfg, default_flow_style=False, sort_keys=False).rstrip())
    print("=" * 50)
    print("Problem")
    print("=" * 50)
    print(yaml.dump({
        "spatial_orbitals": problem.num_spatial_orbitals,
        "electrons": sum(problem.num_particles),
        "qubits": nq,
        "circuit_parameters": transpiled.num_parameters,
        "circuit_depth": transpiled.depth(),
        "statevector_size": _format_bytes(float(2**nq * bpc)),
    }, default_flow_style=False, sort_keys=False).rstrip())
    print("=" * 50)
