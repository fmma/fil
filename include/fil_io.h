#ifndef __FIL_IO_H
#define __FIL_IO_H

#include <cuda_runtime.h>
#include <cufile.h>
#include <ds_file.h>
#include <limits.h>
#include <stdint.h>

struct fil_iter;
struct xal_inode;

struct fil_entry {
	uint64_t dir;
	uint64_t file;
};

struct fil_data {
	struct fil_entry *entries;
	uint64_t n_entries;
	uint64_t index;
};

struct fil_cpu_io {
	uint64_t *slbas;
	uint64_t *elbas;
};

struct fil_file_io {
	char prefix[PATH_MAX];
	char path[PATH_MAX];
	void *buffer;
};

struct fil_gds_io {
	CUfileDescr_t *descr;
	CUfileHandle_t *handle;
	size_t *expected;
	ssize_t *actual;
	cudaStream_t *streams;
};

struct fil_opends_io {
	ds_file_handle_t *handles;
	size_t *expected;
	ssize_t *actual;
	cudaStream_t *streams;
};

int
fil_cpu_submit(struct fil_iter *iter);

int
fil_gpu_submit(struct fil_iter *iter);

int
fil_file_submit(struct fil_iter *iter);

int
fil_gds_async_submit(struct fil_iter *iter);

int
fil_opends_submit(struct fil_iter *iter);

int
fil_opends_async_submit(struct fil_iter *iter);

int
fil_opends_register_entry(struct fil_iter *iter, struct xal_inode *file_inode,
			  uint64_t *mock_fh_out);

int
fil_opends_io_alloc(struct fil_iter *iter);

void
fil_opends_io_free(struct fil_iter *iter);

#endif
