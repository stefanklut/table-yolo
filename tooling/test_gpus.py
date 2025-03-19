import os
import re
import subprocess
import sys

import torch
import torch.version


def get_uuids():
    result = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command 'nvidia-smi -L' failed with exit code {result.returncode}")

    output = result.stdout
    print(output)

    gpu_uuid_pattern = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

    mig_uuid_pattern = re.compile(r"MIG-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

    gpu_uuids = gpu_uuid_pattern.findall(output)

    mig_uuids = mig_uuid_pattern.findall(output)

    return mig_uuids, gpu_uuids


def add_cuda_visible_devices(uuids: list[str]):
    current = os.environ.get("CUDA_VISIBLE_DEVICES")
    if current:
        uuids = current.split(",") + uuids
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ['CUDA_VISIBLE_DEVICES']}")


print(f"Python version: {sys.version}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDNN version: {torch.backends.cudnn.version()}")
print(f"Flash SDP enabled: {torch.backends.cuda.flash_sdp_enabled()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

mig_uuids, gpu_uuids = get_uuids()

if gpu_uuids:
    add_cuda_visible_devices(gpu_uuids)
else:
    print("No GPU devices found.")

if mig_uuids:
    add_cuda_visible_devices(mig_uuids)
else:
    print("No MIG devices found.")

# smoke tests
print(f"torch.randn(1).cuda(): {torch.randn(1).cuda()}")
print(f'torch.empty(2, device="cuda"): {torch.empty(2, device="cuda")}')
