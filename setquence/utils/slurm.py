import os
from typing import Dict, Optional

from setquence.utils.hostlist import expand_hostlist


def slurm_config_to_dict(route: Optional[str] = None, device: str = "autodetect") -> Dict:
    if device == "cuda":
        try:
            device_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
            device = "cuda"
        except KeyError:
            raise KeyError("Could not find any available CUDA devices under CUDA_VISIBLE_DEVICES")
    elif device == "cpu":
        device_ids = list(range(int(os.environ["SLURM_NTASKS"])))
        device = "cpu"
    elif device == "autodetect":
        try:
            return slurm_config_to_dict(device="cuda")
        except KeyError:
            return slurm_config_to_dict(device="cpu")

    try:
        hostnames = expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ["MASTER_ADDR"] = hostnames[0]
        os.environ["MASTER_PORT"] = os.environ.get("SLURM_MASTER_PORT", str(12345))
        size = int(os.environ["SLURM_NTASKS"])
        nodes = int(os.environ["SLURM_NNODES"])
        rank = int(os.environ["SLURM_PROCID"])
        local_device_ids = device_ids[int(rank % int(size / nodes))]

        return {
            "rank": int(os.environ["SLURM_PROCID"]),
            "local_rank": int(os.environ["SLURM_LOCALID"]),
            "size": size,
            "nodes": nodes,
            "nodename": os.environ["SLURMD_NODENAME"],
            "tasks_per_node": int(size / nodes),
            "local_device_rank": int(rank % int(size / nodes)),
            "hostnames": hostnames,
            "device": device,
            "device_ids": device_ids,
            "local_device_ids": [local_device_ids],
        }
    except KeyError:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(12345)

        return {
            "rank": 0,
            "local_rank": 0,
            "size": len(device_ids),
            "nodes": 1,
            "nodename": "localhost",
            "tasks_per_node": 1,
            "local_device_rank": 0,
            "hostnames": ["localhost"],
            "device": device,
            "device_ids": device_ids,
            "local_device_ids": None,
        }
