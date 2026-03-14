#!/bin/bash

set -euo pipefail

sudo apt update
sudo apt install mesa-vulkan-drivers libvulkan1 vulkan-tools intel-gpu-tools mesa-utils # Vulkan driver
sudo apt install intel-opencl-icd clinfo # OpenCL driver

sudo usermod -aG render $USER

echo "reboot required to load Intel iGPU driver (OpenCL and Vulkan)"
