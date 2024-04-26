#!/home/bingxing2/ailab/scxlab0004/.conda/envs/llama_factory/bin/python
import os
import sys

CMD = sys.argv[1]

COMMAND = f'''
echo $SLURM_NODEID
accelerate launch  --rdzv_conf=timeout=30,join_timeout=40 \
    --num_cpu_threads_per_process {os.environ.get('OMP_NUM_THREADS')} \
    --config_file {os.environ.get('SACC_HOME')}/configs/{os.environ.get('SACC_UUID')}/accelerate_$SLURM_NODEID.json {CMD}
'''

run_sh_path = f"{os.environ.get('SACC_HOME')}/configs/{os.environ.get('SACC_UUID')}/run.sh"
with open(run_sh_path, "w") as f:
    f.write(COMMAND)