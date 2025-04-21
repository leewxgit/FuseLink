#ifndef FUSELINK_CUMEM_H
#define FUSELINK_CUMEM_H

#include <cuda.h>
#include <stdint.h>
// 512KB
#define BUFFER_UNIT (512 * 1024)

bool init_cumem();
bool free_cumem(void *ptr);

bool remap_cumem(void *ptr, uint32_t origin_dev_id, uint32_t dev_id);

#endif
