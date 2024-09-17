---
blogpost: true
title: 'Getting to Know Your GPU: A Deep Dive into AMD SMI'
date: 17 Sep 2024
author: Matt Elliott
tags: HPC, Optimization, Performance, System-Tuning
category: Software tools & optimizations
language: English
myst:
  html_meta:
    "description lang=en": "This post introduces AMD System Management Interface (amd-smi), explaining how you can use it to access your GPU’s performance and status data"
    "keywords": "HPC, Optimization, Performance, MI300, libraries"
    "property=og:locale": "en_US"
---

# Getting to Know Your GPU: A Deep Dive into AMD SMI

<span style="font-size:0.7em;">17 Sep, 2024 by {hoverxref}`Matt Elliott<mael>` </span>

For system administrators and power users working with AMD hardware, performance optimization and efficient monitoring of resources is paramount. The AMD System Management Interface command-line tool, `amd-smi`, addresses these needs.

`amd-smi` is a versatile command-line utility designed to manage and monitor AMD hardware, with a primary focus on GPUs. As the future replacement for `rocm-smi`, `amd-smi` is poised to become the primary tool for AMD hardware management across a wide range of devices. For those new to hardware management or transitioning from other tools, `amd-smi` provides an extensive set of features to help optimize AMD hardware usage.

In this blog post, we will provide you with a practical walkthrough of `amd-smi`. We will show you, step-by-step, how to verify the installation of `amd-smi` and how to use its main features: access your AMD GPU’s information and metrics, monitor its performance and running processes in real-time, configure its hardware parameters, inspect your AMD GPU’s topology and memory, and more.

## Understanding system management interfaces

System Management Interfaces (SMIs) are fundamental to modern hardware management and monitoring. These interfaces serve as APIs that provide a standardized way to interact with hardware components. SMIs offer insights into performance and status while allowing for a degree of control.

Users typically interact with SMIs through command-line tools like `amd-smi` or programming libraries, rather than engaging directly with the API. This model enables users to integrate hardware management into their own monitoring and automation frameworks, enabling efficient resource utilization and real-time alerts.

While capabilities might vary between vendors and hardware types, the overarching purpose of SMIs remains consistent: provide users with visibility into and control over their hardware.

## Key features of amd-smi

- **Device information**: Quickly retrieve detailed information about AMD GPUs
- **Performance monitoring**: Real-time monitoring of GPU utilization, memory, temperature, and power consumption
- **Process information**: Identify which processes are using GPUs
- **Configuration management**: Adjust GPU settings like clock speeds and power limits
- **Error reporting**: Monitor and report GPU errors for proactive maintenance

## Getting started with amd-smi

On systems with AMD ROCm™ installed, `amd-smi` should already be available. Verify installation by running this command:

```text
amd-smi version
```

If it's not installed, use the system package manager to install `amd-smi-lib`. For example:

- **Install on Ubuntu**: `sudo apt install amd-smi-lib`
- **Install on RedHat Enterprise Linux (RHEL)**: `sudo dnf install amd-smi-lib`
- **Install on SUSE Linux Enterprise Server (SLES)**: `sudo zypper install amd-smi-lib`

## Basic usage

Here are some common commands and tips for using `amd-smi`.

### List all GPUs

The `amd-smi list` command displays a list of all AMD GPUs in your system, along with basic information like their IDs, PCIe bus addresses, and UUIDs.

```text
$ amd-smi list
GPU: 0
    BDF: 0000:05:00.0
    UUID: afff74a1-0000-1000-8054-e92b0a5d57c8
 
GPU: 1
    BDF: 0000:26:00.0
    UUID: 0aff74a1-0000-1000-805b-ce698de95724
 
GPU: 2
    BDF: 0000:46:00.0
    UUID: 97ff74a1-0000-1000-8065-fa81273af9ce
 
[output truncated]
```

### Display detailed GPU information

The `amd-smi static` command provides comprehensive static information about GPUs, including hardware details, driver versions, and capabilities.

```text
$ amd-smi static
GPU: 0
    ASIC:
        MARKET_NAME: MI300X-O
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a1
        REV_ID: 0x00
        ASIC_SERIAL: 0xAF54E92B0A5D57C8
        OAM_ID: 7
    BUS:
        BDF: 0000:05:00.0
        MAX_PCIE_WIDTH: 16
        MAX_PCIE_SPEED: 32 GT/s
        PCIE_INTERFACE_VERSION: Gen 5
        SLOT_TYPE: OAM
 
[output truncated]
```

### Display detailed GPU metrics

Use `amd-smi metric` to view real-time metrics such as GPU utilization, temperature, power consumption, and memory usage.

```text
$ amd-smi metric
GPU: 0
    USAGE:
        GFX_ACTIVITY: 100 %
        UMC_ACTIVITY: 0 %
        MM_ACTIVITY: N/A
        VCN_ACTIVITY: [0 %, 0 %, 0 %, 0 %]
    POWER:
        SOCKET_POWER: 234 W
        GFX_VOLTAGE: N/A mV
        SOC_VOLTAGE: N/A mV
        MEM_VOLTAGE: N/A mV
        POWER_MANAGEMENT: ENABLED
        THROTTLE_STATUS: UNTHROTTLED
    CLOCK:
        GFX_0:
            CLK: 2102 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 2102 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: DISABLED
        GFX_1:
            CLK: 2101 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 2102 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: DISABLED
        GFX_2:
            CLK: 2107 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 2102 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: DISABLED
[output truncated]
```

### Performance monitoring

The `amd-smi monitor` command displays utilization metrics for GPUs, memory, power, PCIe bandwidth, and more. By default, `amd-smi monitor` outputs 18 metrics for every GPU. Passing in specific arguments limits the types of metrics displayed.

```text
-p, --power-usage            Monitor power usage in Watts
-t, --temperature            Monitor temperature in Celsius
-u, --gfx                    Monitor graphics utilization (%) and clock (MHz)
-m, --mem                    Monitor memory utilization (%) and clock (MHz)
-n, --encoder                Monitor encoder utilization (%) and clock (MHz)
-d, --decoder                Monitor decoder utilization (%) and clock (MHz)
-s, --throttle-status        Monitor thermal throttle status
-e, --ecc                    Monitor ECC single bit, ECC double bit, and PCIe replay error counts
-v, --vram-usage             Monitor memory usage in MB
-r, --pcie                   Monitor PCIe bandwidth in Mb/s
```

For example, to monitor power usage, GPU utilization, temperature, and memory utilization, run `amd-smi monitor -putm`.

```text
$ amd-smi monitor -putm
GPU  POWER  GPU_TEMP  MEM_TEMP  GFX_UTIL  GFX_CLOCK  MEM_UTIL  MEM_CLOCK
  0  182 W     42 °C     41 °C      83 %   1613 MHz       0 %   1230 MHz
  1  143 W     40 °C     39 °C      13 %    358 MHz       0 %   1173 MHz
  2  117 W     41 °C     40 °C       0 %    120 MHz       0 %    900 MHz
  3  116 W     40 °C     38 °C       1 %    134 MHz       0 %    913 MHz
  4  118 W     42 °C     40 °C       0 %    120 MHz       0 %    900 MHz
  5  118 W     39 °C     38 °C       0 %    120 MHz       0 %    900 MHz
  6  116 W     41 °C     41 °C       0 %    120 MHz       0 %    900 MHz
  7  118 W     40 °C     37 °C       0 %    120 MHz       0 %    900 MHz
```

### View running processes

The `amd-smi process` command shows details about processes running on the GPU, including their PIDs, memory usage, and GPU utilization. Running the command with `sudo` includes processes owned by other users.

```text
$ sudo amd-smi process
GPU: 0
    PROCESS_INFO:
        NAME: pt_main_thread
        PID: 207590
        MEMORY_USAGE:
            GTT_MEM: 2.0 MB
            CPU_MEM: 202.0 MB
            VRAM_MEM: 7.3 GB
        MEM_USAGE: 7.5 GB
        USAGE:
            GFX: 0 ns
            ENC: 0 ns
 
GPU: 1
    PROCESS_INFO:
        NAME: pt_main_thread
        PID: 207591
        MEMORY_USAGE:
            GTT_MEM: 2.0 MB
            CPU_MEM: 202.0 MB
            VRAM_MEM: 7.4 GB
        MEM_USAGE: 7.6 GB
 
[output truncated]
```

### Set configurable hardware parameters

The `amd-smi set` command can be used to change hardware parameters such as fan speed, memory and compute partitioning, and power limits. For example, to adjust the power limit of a GPU using the `set` command:

```text
amd-smi set -g 0 -o 650
```

This sets the power limit of GPU 0 to 650 watts. Remember to check the supported power range for your specific GPU model before making adjustments.

Use the `amd-smi reset` command to remove the custom power limit:

```text
amd-smi reset -g 0 -o
```

## Additional capabilities

Run `amd-smi --help` to view the full list of available commands.

```text
AMD-SMI Commands:
 
    version           Display version information
    list              List GPU information
    static            Gets static information about the specified GPU
    firmware (ucode)  Gets firmware information about the specified GPU
    bad-pages         Gets bad page information about the specified GPU
    metric            Gets metric/performance information about the specified GPU
    process           Lists general process information running on the specified GPU
    event             Displays event information for the given GPU
    topology          Displays topology information of the devices
    set               Set options for devices
    reset             Reset options for devices
    monitor           Monitor metrics for target devices
    xgmi              Displays xgmi information of the devices
```

Modifiers are supported with every command to output data as comma-separated values (CSV), JavaScript Object Notation (JSON) or directly to a file.

```text
Command Modifiers:
  --json                   Displays output in JSON format (human readable by default).
  --csv                    Displays output in CSV format (human readable by default).
  --file FILE              Saves output into a file on the provided path (stdout by default).
```

For example, the `--csv` argument passed to `amd-smi process` outputs process information comma-separated values.

```text
$ sudo amd-smi process --csv
gpu,name,pid,gtt_mem,cpu_mem,vram_mem,mem_usage,gfx,enc
0,pt_main_thread,207590,2134016,211795968,7889485824,8103415808,0,0
1,pt_main_thread,207591,2134016,211795968,7923122176,8137052160,0,0
2,pt_main_thread,207589,2134016,211795968,7889575936,8103505920,0,0
3,pt_main_thread,207588,2166784,211763200,7822258176,8036188160,0,0
4,pt_main_thread,207595,2134016,211795968,7889514496,8103444480,0,0
5,pt_main_thread,207590,2134016,211795968,7822381056,8036311040,0,0
6,pt_main_thread,207597,2134016,211795968,7923064832,8136994816,0,0
7,pt_main_thread,207593,2134016,211795968,7889465344,8103395328,0,0
```

The output can be piped to the `column` command to format the values as a table.

```text
$ sudo amd-smi process --csv | column -t -s,
gpu  name            pid     gtt_mem  cpu_mem    vram_mem    mem_usage   gfx  enc
0    pt_main_thread  207590  2134016  211795968  7889485824  8103415808  0    0
1    pt_main_thread  207591  2134016  211795968  7923122176  8137052160  0    0
2    pt_main_thread  207589  2134016  211795968  7889575936  8103505920  0    0
3    pt_main_thread  207588  2166784  211763200  7822258176  8036188160  0    0
4    pt_main_thread  207595  2134016  211795968  7889514496  8103444480  0    0
5    pt_main_thread  207597  2134016  211795968  7822381056  8036311040  0    0
6    pt_main_thread  207594  2134016  211795968  7923064832  8136994816  0    0
7    pt_main_thread  207593  2134016  211795968  7889465344  8103395328  0    0
```

Combining JSON output with the `jq` command can be used to filter results. This example command filters the output from `amd-smi static` to only display VRAM information for the first GPU in the system.

```text
$ amd-smi static --json | jq '.[0]["vram"]'
{
  "type": "HBM",
  "vendor": "N/A",
  "size": {
    "value": 196592,
    "unit": "MB"
  }
}
```

### Display firmware information

Run `amd-smi firmware` to view firmware information for all system GPUs.

```text
$ amd-smi firmware
GPU: 0
    FW_LIST:
        FW 0:
            FW_ID: CP_MEC1
            FW_VERSION: 147
        FW 1:
            FW_ID: CP_MEC2
            FW_VERSION: 147
        FW 2:
            FW_ID: RLC
            FW_VERSION: 64
        FW 3:
            FW_ID: SDMA0
            FW_VERSION: 19
        FW 4:
            FW_ID: SDMA1
            FW_VERSION: 19
        FW 5:
            FW_ID: VCN
            FW_VERSION: 61.13.00.C
        FW 6:
            FW_ID: PSP_SOSDRV
            FW_VERSION: 36.02.4C
        FW 7:
            FW_ID: TA_RAS
            FW_VERSION: 20.00.00.0D
        FW 8:
            FW_ID: TA_XGMI
            FW_VERSION: 20.00.01.13
        FW 9:
            FW_ID: PM
            FW_VERSION: 85.110.0
 
[output truncated]
```

### Inspect memory status

Think of memory like a book with many pages, with each page representing a location where data is stored. If a page becomes "bad," it means that the data stored on that page can't be read or written correctly. Bad pages can occur due to electrical surges, wear and tear, or a variety of other reasons. When a GPU detects a bad page, it marks that page as unusable to prevent errors from spreading. Bad pages can be viewed with the `amd-smi bad-pages` command.

```text
$ amd-smi bad-pages
GPU: 0
    RETIRED: No bad pages found.
    PENDING: No bad pages found.
    UN_RES: No bad pages found.
 
GPU: 1
    RETIRED: No bad pages found.
    PENDING: No bad pages found.
    UN_RES: No bad pages found.
 
[output truncated]
```

### Display GPU topology information

Run `amd-smi topology` to display topology information such as link accessibility, number of hops/relative weight between GPUs, link type, and NUMA bandwidth information.

```text
$ amd-smi topology
ACCESS TABLE:
             0000:05:00.0 0000:26:00.0 0000:46:00.0 0000:65:00.0 0000:85:00.0 0000:a6:00.0 0000:c6:00.0 0000:e5:00.0
0000:05:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:26:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:46:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:65:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:85:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:a6:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:c6:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:e5:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
 
WEIGHT TABLE:
             0000:05:00.0 0000:26:00.0 0000:46:00.0 0000:65:00.0 0000:85:00.0 0000:a6:00.0 0000:c6:00.0 0000:e5:00.0
0000:05:00.0 0            15           15           15           15           15           15           15
0000:26:00.0 15           0            15           15           15           15           15           15
0000:46:00.0 15           15           0            15           15           15           15           15
0000:65:00.0 15           15           15           0            15           15           15           15
0000:85:00.0 15           15           15           15           0            15           15           15
0000:a6:00.0 15           15           15           15           15           0            15           15
0000:c6:00.0 15           15           15           15           15           15           0            15
0000:e5:00.0 15           15           15           15           15           15           15           0
 
[output truncated]
```

### View GPU-to-GPU link metrics

The `amd-smi xgmi` command displays xGMI statistics. xGMI, also known as AMD Infinity Fabric™, provides high-speed connectivity between GPUs.

```text
$ amd-smi xgmi
LINK METRIC TABLE:
       bdf          bit_rate max_bandwidth link_type 05:00.0 26:00.0 46:00.0 65:00.0 85:00.0 a6:00.0 c6:00.0 e5:00.0
GPU0   05:00.0      32 Gb/s  512 Gb/s      XGMI
 Read                                                N/A     1259 KB 1 KB    1 KB    1 KB    1141 KB 9774 KB 1069 KB
 Write                                               N/A     1074 KB 1 KB    1 KB    1 KB    4109 KB 1134 KB 1262 KB
GPU1   26:00.0      32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB     N/A    9808 KB 1022 KB 1 KB    1 KB    1149 KB 1209 KB
 Write                                               0 KB     N/A    1564 KB 1279 KB 1 KB    1 KB    9858 KB 1412 KB
 
[output truncated]
```

### Monitor hardware events

Use the `amd-smi event` command to view event information for all GPUs in the system. After the tool launches, it continues to listen for and display GPU events until stopped. Event types include thermal throttling events, hardware resets, and memory read errors.

```text
$ amd-smi event
EVENT LISTENING:
 
Press q and hit ENTER when you want to stop
```

## Transitioning from rocm-smi

Users familiar with `rocm-smi` will find that `amd-smi` offers similar functionality with some enhancements. Here's a quick comparison of some common commands:

| Task | rocm-smi | amd-smi |
| --- | --- | --- |
| List GPUs | `rocm-smi -i` | `amd-smi list` |
| Show utilization | `rocm-smi` | `amd-smi monitor` |
| Show memory info | `rocm-smi --showmemuse` | `amd-smi monitor -m -v` |
| Show detailed hardware info and settings | `rocm-smi -a` | `amd-smi static` |

While the syntax differs slightly, `amd-smi` generally offers more detailed output and additional features compared to `rocm-smi`.

**Note:** While `rocm-smi` will continue to receive bug fixes and maintenance updates, new features and additional hardware support will be prioritized for `amd-smi`.

## Conclusion

In this blog post we presented a practical deep dive into `amd-smi`, showing you how to use and access its main features and functionalities. Whether you're managing a large-scale computing environment or optimizing a single server, `amd-smi` offers the insights and control needed to maximize the potential of AMD GPUs. To learn more about `amd-smi` and its capabilities, visit the [amd-smi tool documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/amdsmi_cli_readme_link.html).
