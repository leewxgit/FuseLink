# FuseLink

Efficient GPU communication over multiple NICs.

## Build

Hardware:
- RDMA capable NICs (required).
- GPU with GPU Direct RDMA (GDR) capability (required).
- NVLink interconnect (recommended).

Software:
- NCCL v2.19.4
- libibverbs
- CUDA >= 11.8 (To enable GDR and peer access)
- Nvidia driver version 550.120 (Other versions wait for testing)

Build commands

**Build NCCL**
```bash
make -j src.build CUDA_HOME=<your cuda home>
```

**Build FuseLink NCCL Plugin**
```bash
CUDA_HOME=<your cuda home> NCCL_BUILD_DIR=./nccl/build make fl
```

## Usage

Set `NCCL_NET_PLUGIN` environment to `fuselink` and expose `libnccl-net-fuselink.so` to `LD_LIBRARY_PATH`.

```bash
export LD_LIBRARY_PATH=<FuseLink Dir>/build/lib:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=fuselink
```

## Note for OSDI Artifact

We are currently working on additional experiments required by shepherd. Some codes are not stable and tailored for special environments of our testbeds.
