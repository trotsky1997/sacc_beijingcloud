#!/home/bingxing2/ailab/scxlab0003/.conda/envs/llama_factory/bin/python
import sys
import os
import click
import json
import time

uuid = str(time.time())
# JOB_ID_r = os.popen("echo ${SLURM_JOB_ID}")
# JOB_ID 

srcipt_path = os.path.split(os.path.realpath(__file__))[0]
# os.system(f"mkdir {srcipt_path}/configs")
os.system(f"mkdir {srcipt_path}/configs/{uuid}")


def set_hpz_param(zero,gpu_per_nodes):
    deepspeed_config_file = srcipt_path + f'/.deepspeed_zero{zero}.json'
    target_config_file = srcipt_path + f'/configs/{uuid}/.deepspeed.json'
    with open(deepspeed_config_file, 'r') as f:
        deepspeed_config = json.load(f)
    with open(target_config_file, 'x') as f:
        json.dump(deepspeed_config, f, indent=4)

@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--num_nodes', type=int, default=1, help='Number of nodes')
@click.option('--gpu_per_nodes', type=int, default=4, help='Number of GPUs per node')
@click.option('--job_name', type=str, default=f"{os.environ.get('USER')}_job", help='job name')
@click.option('--zero', type=int, default=2, help='Stage of Deepspeed Zero++')
@click.option('--partition', type=str, default="vip_gpu_ailab", help='partition name')
@click.option('--group', type=str, default="ai4phys", help='job name')
@click.argument('**kwargs', nargs=-1, type=click.UNPROCESSED)
def main(num_nodes, gpu_per_nodes, job_name,zero,partition,group,**kwargs):
    set_hpz_param(zero,gpu_per_nodes)
    CMD_START = -1
    for i,j in enumerate(sys.argv[1:]):
        if(j.endswith('.py')):
            CMD_START = i
            break
    CMD = " ".join(sys.argv[CMD_START+1:])
    BASHCOMMAND = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --qos gpugpu
#SBATCH -N {num_nodes}
#SBATCH --gres=gpu:{gpu_per_nodes}
#SBATCH -p {partition}
#SBATCH -A {group}
##SBATCH -w SH-IDC1-10-140-24-123

#SBATCH --output={srcipt_path}/logs/%j.out
#SBATCH --error={srcipt_path}/logs/%j.err

export NCCL_ALGO=Ring
export NCCL_MAX_NCHANNELS=16
export NCCL_MIN_NCHANNELS=16
export NCCL_DEBUG=INFO
export NCCL_TOPO_FILE=/home/bingxing2/apps/nccl/conf/dump.xml
export NCCL_IB_HCA=mx5_0,mlx5_2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=23
export NCCL_IB_RETRY_CNT=7

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_bond_0
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_GID_INDEX=3

export WANDB_MODE=online
export PYTHONUNBUFFERED=1

export PYTHONFAULTHANDLER=1
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_DISTRIBUTED_CUDNN_ENABLED=1
export TORCH_CUDNN_ENABLED=1
export TORCH_DISTRIBUTED_CUDNN_BENCHMARK=1
export TORCH_CUDNN_BENCHMARK=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export OMP_DYNAMIC=TRUE
export OMP_NUM_THREADS=2

export HF_HOME=/home/bingxing2/ailab/group/ai4phys/cache
# export HF_DATASETS_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export TORCH_EXTENSIONS_DIR=/HOME/scw6c7z/run/liuwei/cache/torch_extensions/


export ZERO_STAGE={zero}
export SACC_HOME={srcipt_path}
export SACC_UUID={uuid}
echo $SACC_HOME
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export MASTER_PORT=$(python $SACC_HOME/get_good_port.py)
export GRADIO_SERVER_PORT=10046



export CUDA_HOME='/home/bingxing2/apps/compilers/cuda/cuda-12.1'
# export http_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
# export https_proxy=http://u-cEoRwn:EDvFuZTe@172.16.4.9:3128
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

module load anaconda/2021.11
module load compilers/cuda/12.1
module load cudnn/8.8.1.3_cuda12.x
module load compilers/gcc/9.3.0

source activate py310


srun python $SACC_HOME/gen_ds_hosts.py
srun python $SACC_HOME/gen_accelerate.py
# srun cat /mnt/cache/Chemllm/accelerate_$SLURM_NODEID.json

CMD="{CMD}"

python $SACC_HOME/gen_bash.py "$CMD"


# cd /mnt/cache/Chemllm/LLaMA-Factory

echo $SLURM_JOB_NAME
echo $CMD > $SACC_HOME/logs/$SLURM_JOB_ID.cmd.log


srun bash $SACC_HOME/configs/{uuid}/run.sh > $SACC_HOME/logs/$SLURM_JOB_ID.python.log

'''

    with open(f'{srcipt_path}/configs/{uuid}/run_slurm.sh', 'w') as f:
        f.write(BASHCOMMAND)

    os.system(f'sbatch {srcipt_path}/configs/{uuid}/run_slurm.sh > {srcipt_path}/logs/$SLURM_JOB_ID.log')
    os.system(f'parajobs')

if __name__ == '__main__':
    click.disable_unicode_literals_warning = True
    main()


