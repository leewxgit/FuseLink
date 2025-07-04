#!/bin/bash
BATCH_SZ=2
MODEL_PATH=dummy
#MODEL_NAME=opt_125m
MODEL_NAME=opt_6.7b
#MODEL_NAME=opt_125m
# MODEL_NAME=llama2_7b
# MODEL_NAME=llama2_7b
PRECISION=fp16
INPUT_LEN=512
BLOCK_SZ=32
NUM_DECODE_STEP=1
TP_SZ=2

# calcualte the number of processes
NUM_PROCESS=$((2*TP_SZ))
# run the test
#--prefix $HOME/anaconda3/envs/distserve \
mpirun -np $NUM_PROCESS \
    -H 10.0.13.1:$TP_SZ,10.0.14.1:$TP_SZ \
    --mca btl_tcp_if_include rdma0 \
    -x NCCL_SOCKET_IFNAME=rdma0 \
    -x NCCL_DEBUG=WARN \
    ./FuseLinkServeTest \
    --batch_size $BATCH_SZ \
    --model_path $MODEL_PATH \
    --model_name $MODEL_NAME \
    --precision $PRECISION \
    --input_len $INPUT_LEN \
    --block_size $BLOCK_SZ \
    --num_decoding_step $NUM_DECODE_STEP \
    --tp_size $TP_SZ
#--batch_size 32 --model_path dummy --model_name llama2_7b --precision fp16 --input_len 299 --block_size 32 --num_decoding_step 100 --tp_size 2