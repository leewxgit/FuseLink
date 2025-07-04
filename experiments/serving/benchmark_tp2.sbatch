#!/bin/bash
#SBATCH --job-name=fuselink_serving_tp2
#SBATCH --output=fuselink_serving-tp2%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:2
#SBATCH --partition=normal
#SBATCH --account=fuselink


module purge
module load slurm
module load cuda12.2
module load nvhpc-hpcx-cuda12

# source activate fuselink


# read data path from $1
DATA_PATH=/project/fuselink/rzh/FuseLinkTest/serving/tracegen/trace_1.txt

BATCH_SZ=2
MODEL_PATH=dummy
#MODEL_NAME=opt_125m
MODEL_NAME=opt_30b
#MODEL_NAME=opt_125m
# MODEL_NAME=llama2_7b
# MODEL_NAME=llama2_7b
PRECISION=fp16
INPUT_LEN=2048
BLOCK_SZ=32
NUM_DECODE_STEP=1
TP_SZ=2
# export CUDA_VISIBLE_DEVICES=0
# calcualte the number of processes
NUM_PROCESS=$((2*TP_SZ))
# run the test
# --prefix $HOME/anaconda3/envs/distserve \

cd /project/fuselink/rzh/FuseLinkTest/serving/build/SwiftTransformer/src/examples

echo "$(hostname) slots=$TP_SZ" > hostnames.txt

cat hostnames.txt

srun nvidia-smi

mpirun \
    ./benchmark_all_input_same \
    --model_path $MODEL_PATH \
    $MODEL_NAME \
    $PRECISION \
    diff_block_size_on_large_input_len_small_batch_size_small_decoding_step