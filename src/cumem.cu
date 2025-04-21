#include "cumem.h"

#include <map>
#include "checks.h"
// using map to reverse mapping memory address to device id and memory table index

#define MAX_DEVICES 8
#define MAX_MEM_TABLE_SIZE 16

// global cuda memory tables on devices

struct BackendMemHandle {
  CUmemGenericAllocationHandle cumem_handle;
  CUdeviceptr d_mem;
  bool is_mapped;
};

int dev_num = -1;

CUdevice dev[MAX_DEVICES];
BackendMemHandle mem_table[MAX_DEVICES][MAX_MEM_TABLE_SIZE];
std::map<void *, std::pair<int, int>> mem_to_dev_id_offset;



bool init_cumem() {
  if (dev_num == -1) {
    CUCHECK(cuDeviceGetCount(&dev_num));
  }
  dev_num = std::min(dev_num, MAX_DEVICES);
  if (dev_num == 0) {
    printf("No CUDA devices found\n");
    return false;
  }
  for (int i = 0; i < dev_num; i++) {
    CUCHECK(cuDeviceGet(&dev[i], i));
    // allocate memory handle for each device
    for (int j = 0; j < MAX_MEM_TABLE_SIZE; j++) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev[i];
        prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
        // align
        size_t granularity = 0;
        CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size_t aligned_sz = ((BUFFER_UNIT + granularity - 1) / granularity) * granularity;
        CUCHECK(cuMemCreate(&mem_table[i][j].cumem_handle, aligned_sz, &prop, 0));
        mem_table[i][j].d_mem = NULL;
        mem_table[i][j].is_mapped = false;
    }
  }
}

// free all associated resources
bool free_cumem(void *ptr) {
  auto it = mem_to_dev_id_offset.find(ptr);
  if (it == mem_to_dev_id_offset.end()) {
    printf("No mapping found for the pointer\n");
    return false;
  }
  CUCHECK(cuMemUnmap((CUdeviceptr)ptr, BUFFER_UNIT));
  CUCHECK(cuMemRelease(mem_table[it->second.first][it->second.second].cumem_handle));
  mem_to_dev_id_offset.erase(it);
  return true;
}

bool remap_cumem(void *ptr, uint32_t origin_dev_id, uint32_t dev_id) {
  // check the current mapping
  auto it = mem_to_dev_id_offset.find(ptr);
  if (it != mem_to_dev_id_offset.end()) {
    // release the old mapping
    CUCHECK(cuMemUnmap((CUdeviceptr)ptr, BUFFER_UNIT));
    // remap the memory
    CUCHECK(cuMemMap((CUdeviceptr)ptr, BUFFER_UNIT, 0, mem_table[it->second.first][it->second.second].cumem_handle, 0));
    // update mappig
    mem_to_dev_id_offset[ptr] = {dev_id, it->second.second};

    return true;
  } else { // not yet mapped
    // find the available slot
    for (int i = 0; i < MAX_MEM_TABLE_SIZE; i++) {
      if (!mem_table[dev_id][i].is_mapped) {
        // map the memory
        CUCHECK(cuMemMap((CUdeviceptr)ptr, BUFFER_UNIT, 0, mem_table[dev_id][i].cumem_handle, 0));
        mem_table[dev_id][i].is_mapped = true;
        mem_table[dev_id][i].d_mem = (CUdeviceptr)ptr;
        mem_to_dev_id_offset[ptr] = {dev_id, i};

        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = origin_dev_id;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess(mem_table[dev_id][i].d_mem, BUFFER_UNIT, &accessDesc, 1));
        return true;
      }
    }
  }
  printf("No available slot for remapping\n");
  return false;
}
