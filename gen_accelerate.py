#!/home/bingxing2/ailab/scxlab0004/.conda/envs/llama_factory/bin/python
import os
import sys
import json
import socket

zero3_flag = ',\n"zero3_init_flag": true' if os.environ.get("ZERO_STAGE") == "3" else ""


conf = json.loads('''
{
  "compute_environment": "LOCAL_MACHINE",
  "debug": false,
  "deepspeed_config": {
    "deepspeed_config_file": "'''+os.environ.get("SACC_HOME")+f'''/configs/{os.environ.get('SACC_UUID')}/.deepspeed.json",
    "deepspeed_hostfile": "'''+os.environ.get("SACC_HOME")+f'''/configs/{os.environ.get('SACC_UUID')}/ds_hosts",
    "deepspeed_multinode_launcher": "standard"'''+zero3_flag+'''
  },
  "distributed_type": "DEEPSPEED",
  "downcast_bf16": "no",
  "machine_rank": 0,
  "main_training_function": "main",
  "main_process_ip": "",
  "main_process_port": 23456,
  "tpu_env": [],
  "tpu_use_cluster": false,
  "tpu_use_sudo": false,
  "use_cpu": false,
  "num_machines": 2,
  "num_processes": 8,
  "rdzv_backend": "static",
  "same_network": true
}
''')



conf['machine_rank'] = int(os.environ.get("SLURM_NODEID"))
conf['num_machines'] = int(os.environ.get("SLURM_JOB_NUM_NODES"))
conf['num_processes'] = int(os.environ.get("CUDA_VISIBLE_DEVICES").split(",").__len__()) * int(os.environ.get("SLURM_JOB_NUM_NODES"))
conf['main_process_ip'] = socket.gethostbyname(os.environ.get("MASTER_ADDR"))
conf['main_process_port'] = int(os.environ.get("MASTER_PORT"))

with open(f'{os.environ.get("SACC_HOME")}/configs/{os.environ.get("SACC_UUID")}/accelerate_{os.environ.get("SLURM_NODEID")}.json', 'w') as f:
    json.dump(conf, f, indent=4)

