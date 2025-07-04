#!/bin/bash
BATCH_SZ=1
MODEL_PATH=dummy
MODEL_NAME=opt_125m
#MODEL_NAME=opt_6.7b
PRECISION=fp32
INPUT_LEN=1024
BLOCK_SZ=32
NUM_DECODE_STEP=100
TP_SZ=2

# calcualte the number of processes
NUM_PROCESS=$((TP_SZ))
# run the test
#--prefix $HOME/anaconda3/envs/distserve \
mpirun -np $NUM_PROCESS \
    -H 10.0.14.1:$TP_SZ \
    --mca btl_tcp_if_include rdma0 \
    -x NCCL_SOCKET_IFNAME=rdma0 \
    -x NCCL_DEBUG=INFO \
    ./FuseLinkServeTest \
    --batch_size $BATCH_SZ \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --precision $PRECISION \
    --input_len $INPUT_LEN \
    --block_size $BLOCK_SZ \
    --num_decoding_step $NUM_DECODE_STEP \
    --tp_size $TP_SZ