# Run using: 
# python -m torch.distributed.run --nproc_per_node=8 demo_torch_error.py
import torch.distributed as dist


def main():
    dist.init_process_group(backend="gloo", init_method="env://")
    print(f"This a coding error ... {1 / 0}")


if __name__ == "__main__":
    main()
