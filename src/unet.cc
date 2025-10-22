#include "unet.h"
#include "checks.h"

void UnetMemRegionInit(int nGPUs, void* base_addr, size_t size, int src_dev, int buffer_dev, struct UnetMemRegion *unmr, bool is_cuda_mem) {
  unmr->nrefs = 0;
  pthread_mutex_init(&unmr->lock, NULL);

  if (!is_cuda_mem) {
    // allocate cpu memory
    unmr->addr[MAX_GPU_NUM] = (CUdeviceptr) base_addr;
    unmr->sz = size;
    return;
  }

  size_t granularity = 0;
  size_t aligned_sz = 0;
  aligned_sz = ((size + granularity - 1) / granularity) * granularity;
  unmr->sz = aligned_sz;

  for (int i = 0; i < nGPUs; i++) {
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = i;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.allocFlags.gpuDirectRDMACapable = 1;
    
    FL_CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    aligned_sz = ((size + granularity - 1) / granularity) * granularity;
    unmr->sz = aligned_sz;

    FL_CUCHECK(cuMemCreate(&unmr->hdl[i], aligned_sz, &prop, 0));
    for (uint j = 0; j < MAX_NIC_NUM; j++) {
      unmr->mr[i][j] = NULL;
    }
    
    // create address range
    FL_CUCHECK(cuMemAddressReserve(&unmr->addr[i], aligned_sz, 0, 0, 0));
    FL_CUCHECK(cuMemMap(unmr->addr[i], aligned_sz, 0, unmr->hdl[i], 0));

    // TODO: may need to enable access to this memory region
  }
  CUmemGenericAllocationHandle origin_hdl;
  FL_CUCHECK(cuMemRetainAllocationHandle(&origin_hdl, base_addr));
  FL_CUCHECK(cuMemUnmap((CUdeviceptr) base_addr, aligned_sz));
  FL_CUCHECK(cuMemRelease(origin_hdl));

  printf("map buffer addr %p, size %zu, handle %p bufferdev %d\n", base_addr, aligned_sz, (void*)unmr->hdl[buffer_dev], buffer_dev);
  FL_CUCHECK(cuMemMap((CUdeviceptr) base_addr, aligned_sz, 0, unmr->hdl[buffer_dev], 0));

  // enable access to this memory region
  CUmemAccessDesc accessDesc = {};
  accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  accessDesc.location.id = src_dev;
  accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  FL_CUCHECK(cuMemSetAccess((CUdeviceptr) base_addr, aligned_sz, &accessDesc, 1));

  return;
}

void UnetMemRegionDestroy(struct UnetMemRegion *unmr) {
  // return
  return;
}
