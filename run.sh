test=../nccl-tests/build/sendrecv_perf
# if exist $3, use $3 as BUF, else use 1048576(1M) as BUF
if [ -z $3 ]; then
    BUF=1048576
else
    BUF=$3
fi
# on gpu 13 and 14
# mpirun -np 2 \
#     --host 10.0.13.1,10.0.14.1 \
#     -mca btl_tcp_if_include rdma0,rdma1 \
#     -x LD_LIBRARY_PATH=/usr/local/lib:/home/zhenghang/work_dir/gpu_multipath/build/lib:/home/zhenghang/work_dir/nccl/build/lib \
#     -x NCCL_NET_PLUGIN=$1 \
#     -x NCCL_DEBUG=$2 \
#     -x NCCL_SOCKET_IFNAME=rdma0 \
#     -x NCCL_IB_GID_INDEX=3 \
#     -x NCCL_CUMEM_ENABLE=1 \
#     -x NCCL_NET_SHARED_BUFFERS=0 \
#     -x NCCL_MAX_NCHANNELS=2 \
#     -x NCCL_MIN_NCHANNELS=2 \
#     -x NCCL_DEBUG_SUBSYS=INIT,NET,P2P,ALLOC \
#     -x NCCL_BUFFSIZE=${BUF} \
#     ${test} \
#     -b 16M \
#     -e 16M \
#     -N $4 \
#     -g 1

# on tacc gpu 02 and 03
# mpirun -np 2 \
#     --host gpu03,gpu05 \
#     --prefix /home/zhenghang/miniconda3/envs/aicc \
#     -mca btl_tcp_if_include eno1 \
#     -x LD_LIBRARY_PATH=/usr/local/lib:/home/zhenghang/work_dir/gpu_multipath/build/lib:/home/zhenghang/work_dir/nccl/build/lib:$CONDA_PREFIX/lib \
#     -x NCCL_NET_PLUGIN=$1 \
#     -x NCCL_DEBUG=$2 \
#     -x NCCL_SOCKET_IFNAME=eno1 \
#     -x NCCL_IB_GID_INDEX=3 \
#     -x NCCL_CUMEM_ENABLE=1 \
#     -x NCCL_NET_SHARED_BUFFERS=0 \
#     -x NCCL_MAX_NCHANNELS=1 \
#     -x NCCL_MIN_NCHANNELS=1 \
#     -x NCCL_NCHANNELS_PER_NET_PEER=1 \
#     -x NCCL_DEBUG_SUBSYS=INIT,NET,P2P,ALLOC \
#     -x NCCL_BUFFSIZE=${BUF} \
#     ${test} \
#     -b 512K \
#     -e 512K \
#     -n 5 \
#     -g 1
# on dgpu13 and dgpu14
mpirun -np 2 \
    --host dgpu13,dgpu14 \
    --prefix /home/zhenghang/miniconda3/envs/fuselink \
    -mca btl_tcp_if_include enp1s0f0 \
    -x LD_LIBRARY_PATH=/usr/local/lib:/home/zhenghang/work_dir/fuselink/build/lib:/home/zhenghang/work_dir/nccl/build/lib:$CONDA_PREFIX/lib \
    -x NCCL_NET_PLUGIN=$1 \
    -x NCCL_DEBUG=$2 \
    -x NCCL_CUMEM_ENABLE=1 \
    -x NCCL_MIN_NCHANNELS=1 \
    -x NCCL_MAX_NCHANNELS=1 \
    -x NCCL_IB_HCA=mlx5_1 \
    -x NCCL_NCHANNELS_PER_NET_PEER=1 \
    -x NCCL_P2P_NET_CHUNKSIZE=524288 \
    -x NCCL_DEBUG_SUBSYS=INIT,NET,P2P,ALLOC \
    ${test} \
    -b 128M \
    -e 128M \
    -n 5 \
    -g 1

    # -x NCCL_NET_SHARED_BUFFERS=0 \
    #     -x NCCL_IB_GID_INDEX=3 \
