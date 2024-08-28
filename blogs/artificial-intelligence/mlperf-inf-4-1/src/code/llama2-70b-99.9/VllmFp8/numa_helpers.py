from collections import defaultdict
import subprocess
import logging
import itertools

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__file__)

DEVICE_INDEX_STR = "Device Index"
DEVICE_TYPE_STR = "Device Type"
INTER_DEVICE_ACCESS_STR = "Inter-Device Access"
INTER_DEVICE_NUMA_STR = "Inter-Device Numa Distance"


class Device:
    def __init__(self, idx, type):
        self.idx = int(idx)
        self.type = type

    def is_gpu(self):
        return self.type == "GPU"

    def is_cpu(self):
        return self.type == "CPU"


def parse_line(line: str):
    return line.split(":")[-1].strip()


def get_devices(lines):
    devices = []
    for idx, line in enumerate(lines):
        if DEVICE_INDEX_STR in line:
            device_idx = parse_line(line)
            if not DEVICE_TYPE_STR in lines[idx + 1]:
                raise Exception(
                    f"Expected '{DEVICE_TYPE_STR}' after '{DEVICE_INDEX_STR}'"
                )
            type = parse_line(lines[idx + 1])
            devices.append(Device(device_idx, type))
    logging.debug(f"Devices: {len(devices)}")
    return devices


def get_numa_matrix(lines, dim: int):
    def str_list_to_int_list(list):
        return [int(num) for num in list]

    numa_info_trimmed = [line.strip().split()[1:] for line in lines if line.strip()]
    numa_matrix = [str_list_to_int_list(line) for line in numa_info_trimmed[1:]]
    assert len(numa_matrix) == dim
    for line in numa_matrix:
        assert len(line) == dim
    logging.debug(f"Numa matrix: {numa_matrix}")
    return numa_matrix


def find_closest_cpus_to_gpu(numa_matrix, gpu_idx: int, cpu_indices):
    smallest_dist = 1000000
    closest_cpus = []
    for idx, dist in enumerate(numa_matrix[gpu_idx]):
        if idx not in cpu_indices:
            continue
        if dist == smallest_dist:
            closest_cpus.append(idx)
        if dist < smallest_dist:
            smallest_dist = dist
            closest_cpus = [idx]
    return closest_cpus


def create_gpu_to_cpu_mapping(numa_matrix, devices):
    gpu_to_cpu_map = {}
    gpu_indices = [device.idx for device in devices if device.is_gpu()]
    cpu_indices = [device.idx for device in devices if device.is_cpu()]
    logging.debug(f"gpu_indices {gpu_indices}")
    logging.debug(f"cpu_indices {cpu_indices}")
    for gpu_idx in gpu_indices:
        cpus = find_closest_cpus_to_gpu(numa_matrix, gpu_idx, cpu_indices)
        gpu_to_cpu_map[gpu_idx] = cpus
    smallest_gpu_idx = min(gpu_indices)
    smallest_cpu_idx = min(cpu_indices)
    gpu_to_cpu_map_aligned = {}
    for k, v in gpu_to_cpu_map.items():
        gpu_to_cpu_map_aligned[k - smallest_gpu_idx] = [
            cpu_idx - smallest_cpu_idx for cpu_idx in v
        ]
    logging.debug(f"GPU to CPU map {gpu_to_cpu_map_aligned}")
    return gpu_to_cpu_map_aligned

def run_rocm_bandwidth_test():
    p = subprocess.run(
        ["/usr/local/bin/rocm-bandwidth-test", "-t"], capture_output=True, text=True
    )
    return p.stdout

def get_gpu_to_cpu_mapping():
    input = run_rocm_bandwidth_test()
    device_info = input.split(INTER_DEVICE_ACCESS_STR)[0].split("\n")
    devices = get_devices(device_info)
    numa_info = input.split(INTER_DEVICE_NUMA_STR)[-1].split("\n")
    numa_matrix = get_numa_matrix(numa_info, len(devices))
    return create_gpu_to_cpu_mapping(numa_matrix, devices)

from numa import schedule, memory, info
import os
import itertools

def set_affinity_by_device(device_id):
    gpu_to_cpu_map = get_gpu_to_cpu_mapping()
    nodes = gpu_to_cpu_map[device_id]

    log.info(f"GPU: {device_id}")
    log.info(f"Nearest nodes = {nodes}")
    schedule.run_on_nodes(*nodes)
    memory.set_membind_nodes(*nodes)

    assert schedule.get_affinitive_nodes().sort() == nodes.sort()
    assert memory.get_membind_nodes().sort() == nodes.sort()

def log_device_affinities(device_id):
    log.info(f"GPU: {device_id}")
    log.info(f"Bound node = {memory.get_membind_nodes().sort()}")

def set_affinity_by_device_and_pid(device_id, pid):
    gpu_to_cpu_map = get_gpu_to_cpu_mapping()
    nodes = gpu_to_cpu_map[device_id]

    numa_hw_info = info.numa_hardware_info()
    cpu_ids = list(itertools.chain.from_iterable([numa_hw_info['node_cpu_info'][n] for n in nodes]))
    log.info(f"GPU: {device_id}")
    log.info(f"CPU ids: {cpu_ids}")
    log.info(f"Nearest nodes = {nodes}")
    schedule.run_on_cpus(pid, *cpu_ids)
    bound_nodes = memory.get_membind_nodes()
    if all(n not in nodes for n in bound_nodes):
        log.error("All numa nodes out of supported range for device{device_id}")

    assert schedule.get_affinitive_cpus(pid).sort() == cpu_ids.sort()
    assert memory.get_membind_nodes().sort() == nodes.sort()
