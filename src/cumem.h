#ifndef FUSELINK_CUMEM_H
#define FUSELINK_CUMEM_H

#include "cudawrap.h"
#include <stdint.h>
// 512KB
#define BUFFER_UNIT (2 * 1024 * 1024) // 2MB

bool init_cumem();
bool free_cumem(void *ptr);

bool remap_cumem(void *ptr, uint32_t origin_dev_id, uint32_t dev_id);

int register_cumem(void *ptr, size_t size);

int ptr2dev(void *ptr);

int show_cumem_usage();

int cumem_dev(void *ptr);

#endif
