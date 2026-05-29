# VQE Experiment Runner

## Usage

```bash
python3 vqe.py --config configs/lih_631g_uccsd_slsqp_cpu.yaml
```

Example configs:

- CPU backend: `configs/lih_631g_uccsd_slsqp_cpu.yaml`
- Single GPU backend: `configs/lih_631g_uccsd_slsqp_gpu.yaml`
- Multi-GPU backend: `configs/lih_631g_uccsd_slsqp_multi_gpu.yaml`
- Efficient SU2 ansatz: `configs/lih_631g_efficient_su2_slsqp_gpu.yaml`

To disable MLflow logging, add `--no-mlflow`:

```bash
python vqe.py --config configs/lih_631g_uccsd_slsqp_cpu.yaml --no-mlflow
```

## Inspecting results with MLflow

By default, runs are logged to a local `mlruns/` directory under the `VQE` experiment. Launch the MLflow UI with:

```bash
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0 --port 5000
```

Then open <http://localhost:5000/> to browse runs, compare energy convergence curves, and inspect logged parameters.
