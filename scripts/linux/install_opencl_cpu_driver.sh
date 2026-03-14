#!/bin/bash

set -euo pipefail

install_clinfo() {
    sudo apt install -yq clinfo
}

install_pocl_fallback() {
    echo "Falling back to Ubuntu POCL OpenCL CPU runtime"
    sudo apt update
    sudo apt install -yq pocl-opencl-icd ocl-icd-libopencl1
    install_clinfo
}

try_install_intel_oneapi() {
    local keyring=/usr/share/keyrings/oneapi-archive-keyring.gpg
    local repo_file=/etc/apt/sources.list.d/oneAPI.list
    local repo_url=https://apt.repos.intel.com/oneapi
    local key_url=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB

    tmp_key=$(mktemp)
    trap 'rm -f "$tmp_key"' RETURN

    curl -fsSL "$key_url" -o "$tmp_key" || return 1
    gpg --dearmor < "$tmp_key" | sudo tee "$keyring" > /dev/null || return 1
    echo "deb [signed-by=$keyring] $repo_url all main" | sudo tee "$repo_file" > /dev/null
    sudo apt update || return 1
    sudo apt install -yq intel-oneapi-runtime-opencl || return 1
    install_clinfo
}

if ! try_install_intel_oneapi; then
    install_pocl_fallback
fi

clinfo -l
