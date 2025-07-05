#!/bin/bash
#SBATCH --job-name=test_bw
#SBATCH --output=test_bw-%j.out
#SBATCH --error=test_bw-%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=normal
#SBATCH --account=fuselink


module purge
module load slurm
module load cuda12.2
module load nvhpc-hpcx-cuda12

test=/project/fuselink/nccl-tests/build/sendrecv_perf

BUF=4194304

# cd to the directory of sbatch file
cd /project/fuselink/gpu_multipath

# get hostname
srun hostname > hostnames.txt

mpirun -np 2 \
    --hostfile hostnames.txt \
    -x LD_LIBRARY_PATH=/project/fuselink/fuselink/build/lib:/project/fuselink/nccl/build/lib:$LD_LIBRARY_PATH \
    -x NCCL_NET_PLUGIN=fuselink \
    -x NCCL_DEBUG=INFO \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_CUMEM_ENABLE=1 \
    -x NCCL_NET_SHARED_BUFFERS=0 \
    -x NCCL_FUSELINK_DEGREE=2 \
    -x NCCL_FUSELINK_STEP=1 \
    -x NCCL_MAX_NCHANNELS=4 \
    -x NCCL_MIN_NCHANNELS=4 \
    -x NCCL_DEBUG_SUBSYS=INIT,NET,P2P,ALLOC,ALL \
    -x NCCL_BUFFSIZE=${BUF} \
    -x NCCL_IB_HCA='=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11' \
    -x NCCL_NCHANNELS_PER_NET_PEER=4 \
    -x NCCL_GDR_FLUSH_DISABLE=1 \
    -x NCCL_NET_GDR_LEVEL=SYS \
    ${test} \
    -b 1G \
    -e 1G \
    -g 1
    
# -x NCCL_IB_HCA='=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11' \