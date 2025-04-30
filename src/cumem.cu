#include "cumem.h"

#include <map>
#include "checks.h"
// using map to reverse mapping memory address to device id and memory table index
#include <assert.h>

#define MAX_DEVICES 8
#define MAX_MEM_TABLE_SIZE 256

// global cuda memory tables on devices

struct BackendMemHandle {
  CUmemGenericAllocationHandle cumem_handle;
  CUdeviceptr d_mem;
  bool is_mapped;
};

int dev_num = -1;
size_t granularity = 0;

CUdevice dev[MAX_DEVICES];
BackendMemHandle mem_table[MAX_DEVICES][MAX_MEM_TABLE_SIZE];
std::map<void *, std::pair<int, int>> mem_to_dev_id_offset;
std::map<void *, size_t> ptr_to_size; // cache ptr size
std::map<void *, int> ptr_to_dev; // cache ptr dev


bool init_cumem() {
  if (dev_num == -1) {
    FL_CUDACHECK(cudaGetDeviceCount(&dev_num));
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
        prop.allocFlags.gpuDirectRDMACapable = 1;
        // align
        CUCHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        size_t aligned_sz = ((BUFFER_UNIT + granularity - 1) / granularity) * granularity;
        CUCHECK(cuMemCreate(&mem_table[i][j].cumem_handle, aligned_sz, &prop, 0));
        mem_table[i][j].d_mem = NULL;
        mem_table[i][j].is_mapped = false;
    }
  }
  return true;
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

int register_cumem(void *ptr, size_t size) {
  // register a gpu memory
  int dev = -1;

  CUCHECK(cuPointerGetAttribute(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr) ptr));
  if (dev < 0 ) return -1; // only support GPU memory

  CUdeviceptr base_ptr;
  size_t base_size;
  printf("ptr %p\n", ptr);
  CUCHECK(cuMemGetAddressRange(&base_ptr, &base_size, (CUdeviceptr) ptr));
  printf("ptr %p base_ptr: %p, base_size: %zu\n", ptr, (void *)base_ptr, base_size);
  // align ptr to 512KB
  // void *aligned_ptr = (void *)(((uintptr_t)ptr + BUFFER_UNIT - 1) / BUFFER_UNIT * BUFFER_UNIT);
  // void *aligned_ptr = (void *) base_ptr;
  // aligned_ptr = (void *)(((uintptr_t)aligned_ptr + BUFFER_UNIT - 1) / BUFFER_UNIT * BUFFER_UNIT);

  // assert((uintptr_t)aligned_ptr == (uintptr_t) base_ptr);

  // check if the ptr is already registered
  auto it = ptr_to_size.find((void *)base_ptr);
  if (it != ptr_to_size.end())
    return 0;
  
  // old allocation handle
  CUmemGenericAllocationHandle old_handle;
  CUCHECK(cuMemRetainAllocationHandle(&old_handle, (void*) ptr));

  // if (!old_handle) {
  //   printf("Failed to retain allocation handle\n");
  //   return -1;
  // }

  CUCHECK(cuMemUnmap((CUdeviceptr) base_ptr, base_size));
  // CUCHECK(cuMemRelease(old_handle));
  // CUCHECK(cuMemAddressFree(base_ptr, base_size));

  // register memory to mem pool
  int step_size = BUFFER_UNIT;
  int aligned_size = ((base_size + BUFFER_UNIT - 1) / BUFFER_UNIT) * BUFFER_UNIT;
  int steps = aligned_size / step_size;

  // go for each step
  uintptr_t start_ptr = (uintptr_t) base_ptr;
  for (int i = 0; i < steps; i++) {
    // find a slot
    CUdeviceptr tmp_ptr = (CUdeviceptr) start_ptr;
    // CUCHECK(cuMemAddressReserve(&tmp_ptr, step_size, granularity, (CUdeviceptr) start_ptr, dev));
    assert(tmp_ptr == (CUdeviceptr) start_ptr);
    bool found = false;
    for (int j = 0; j < MAX_MEM_TABLE_SIZE; j++) {
      if (!mem_table[dev][j].is_mapped) {
        found = true;
        // map the memory
        printf("map %p step %d/%d step size %d cumem handle %p\n", (void *)tmp_ptr, i, steps, step_size, (void *)mem_table[dev][j].cumem_handle);
        CUCHECK(cuMemMap((CUdeviceptr)tmp_ptr, step_size, 0, mem_table[dev][j].cumem_handle, 0));
        // CUCHECK(cuMemMap((CUdeviceptr)tmp_ptr, step_size, (uint64_t) tmp_ptr - (uint64_t)base_ptr, old_handle, 0));
        mem_table[dev][j].is_mapped = true;
        mem_table[dev][j].d_mem = (CUdeviceptr)tmp_ptr;
        mem_to_dev_id_offset[(void *)tmp_ptr] = {dev, j};
        
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = dev;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess(mem_table[dev][j].d_mem, BUFFER_UNIT, &accessDesc, 1));
        break;
      }
    }
    if (!found) {
      printf("No available slot for registering memory\n");
      return -1;
    }
    // update the start pointer
    start_ptr += step_size;
  }
  return 0;
}

int ptr2dev(void *ptr) {
  auto it = mem_to_dev_id_offset.find(ptr);
  if (it == mem_to_dev_id_offset.end()) {
    printf("No mapping found for the pointer\n");
    return -1;
  }
  return it->second.first;
}


int show_cumem_usage() {
  for (int i = 0; i < dev_num; i++) {
    for (int j = 0; j < MAX_MEM_TABLE_SIZE; j++) {
      printf("Device %d, slot %d: %p %d\n", i, j, (void *)mem_table[i][j].d_mem, mem_table[i][j].is_mapped);
    }
  }
  return 0;
}

int cumem_dev(void *ptr) {
  int dev = -1;
  CUCHECK(cuPointerGetAttribute(&dev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr));
  return dev;
}