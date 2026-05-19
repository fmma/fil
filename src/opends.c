#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>

#include <ds_file.h>
#include <ds_file_async.h>
#include <fs_mock.h>
#include <homi_types.h>
#include <libxal.h>
#include <libxnvme.h>

#define ELAPSED(s, e) \
	((double)((e).tv_sec - (s).tv_sec) + (double)((e).tv_nsec - (s).tv_nsec) / 1e9)

int
fil_opends_submit(struct fil_iter *iter)
{
	struct fil_entry entry;
	struct fil_dev *device = iter->devs[0];
	struct timespec start, end;
	uint32_t buf_id;
	void *buffer;
	uint64_t nbytes;
	int mock_fh;
	ds_file_handle_t fh;
	ds_file_error_t derr;
	ssize_t got;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		mock_fh = (int)entry.file;
		if (fs_mock_get_size(mock_fh, &nbytes) < 0) {
			fprintf(stderr, "fs_mock_get_size(%d): out of range\n", mock_fh);
			return EIO;
		}

		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];

		iter->output->buf_len[buf_id] = nbytes;
		iter->output->labels[buf_id] = (uint32_t)entry.dir;
		iter->stats->bytes += nbytes;
		iter->stats->io++;

		derr = ds_file_handle_register(&fh, mock_fh);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_handle_register(%d): %s\n",
				mock_fh, ds_file_op_status_error(derr.err));
			return derr.err;
		}

		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += ELAPSED(start, end);

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		got = ds_file_read(fh, buffer, nbytes, 0, 0);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += ELAPSED(start, end);

		ds_file_handle_deregister(fh);

		if (got < 0) {
			fprintf(stderr, "ds_file_read(%d): %s\n", mock_fh,
				ds_file_op_status_error((ds_file_op_error_t)(-got)));
			return EIO;
		}
		if ((uint64_t)got != nbytes) {
			fprintf(stderr,
				"ds_file_read(%d) short: expected %lu, got %ld\n",
				mock_fh, nbytes, got);
			return EIO;
		}

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	}
	return 0;
}

int
fil_opends_async_submit(struct fil_iter *iter)
{
	struct fil_entry entry;
	struct fil_dev *device = iter->devs[0];
	struct fil_opends_io *io = iter->opends_io;
	struct timespec start, end;
	uint32_t buf_id;
	void *buffer;
	int mock_fh;
	ds_file_error_t derr;
	off_t offset = 0;
	int err;
	uint64_t buf_start = device->buf;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		uint64_t nbytes;
		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		mock_fh = (int)entry.file;
		if (fs_mock_get_size(mock_fh, &nbytes) < 0) {
			fprintf(stderr, "fs_mock_get_size(%d): out of range\n", mock_fh);
			return EIO;
		}

		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];

		iter->output->buf_len[buf_id] = nbytes;
		iter->output->labels[buf_id] = (uint32_t)entry.dir;
		iter->stats->bytes += nbytes;
		iter->stats->io++;

		derr = ds_file_handle_register(&io->handles[i], mock_fh);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_handle_register(%d): %s\n",
				mock_fh, ds_file_op_status_error(derr.err));
			return derr.err;
		}

		io->expected[i] = nbytes;
		io->actual[i] = 0;

		derr = ds_file_read_async(io->handles[i], buffer,
					  &io->expected[i], &offset, &offset,
					  &io->actual[i], io->streams[i]);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_read_async(%d): %s\n",
				mock_fh, ds_file_op_status_error(derr.err));
			return derr.err;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->prep_time += ELAPSED(start, end);

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		err = cudaStreamSynchronize(io->streams[i]);
		if (err) {
			fprintf(stderr, "cudaStreamSynchronize: %d\n", err);
			return err;
		}
		if (io->actual[i] < 0) {
			fprintf(stderr, "ds_file_read_async failed: %s\n",
				ds_file_op_status_error(
					(ds_file_op_error_t)(-io->actual[i])));
			return EIO;
		}
		if ((size_t)io->actual[i] != io->expected[i]) {
			fprintf(stderr,
				"ds_file_read_async short: expected %lu, got %ld\n",
				io->expected[i], io->actual[i]);
			return EIO;
		}
	}
	clock_gettime(CLOCK_MONOTONIC_RAW, &end);
	iter->stats->io_time += ELAPSED(start, end);

	if (iter->opts->verify) {
		for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
			size_t nbytes = io->expected[i];
			void *async_dev =
				device->buffers[(buf_start + i) % device->n_buffers];
			ssize_t got;

			if (nbytes > io->verify_buf_size) {
				fprintf(stderr,
					"verify: entry %u nbytes %lu exceeds buf %lu\n",
					i, (unsigned long)nbytes,
					(unsigned long)io->verify_buf_size);
				return EIO;
			}

			err = cudaMemcpy(io->verify_host_async, async_dev,
					 nbytes, cudaMemcpyDeviceToHost);
			if (err) {
				fprintf(stderr,
					"verify: cudaMemcpy async D2H: %d\n", err);
				return err;
			}

			got = ds_file_read(io->handles[i], io->verify_dev_buf,
					   nbytes, 0, 0);
			if (got < 0) {
				fprintf(stderr,
					"verify: ds_file_read(%d): %s\n",
					(int)iter->data->entries[
						(iter->data->index - iter->opts->batch_size + i)
						% iter->data->n_entries].file,
					ds_file_op_status_error(
						(ds_file_op_error_t)(-got)));
				return EIO;
			}
			if ((size_t)got != nbytes) {
				fprintf(stderr,
					"verify: sync short read: expected %lu, got %ld\n",
					(unsigned long)nbytes, (long)got);
				return EIO;
			}

			err = cudaMemcpy(io->verify_host_sync, io->verify_dev_buf,
					 nbytes, cudaMemcpyDeviceToHost);
			if (err) {
				fprintf(stderr,
					"verify: cudaMemcpy sync D2H: %d\n", err);
				return err;
			}

			if (memcmp(io->verify_host_async, io->verify_host_sync,
				   nbytes) != 0) {
				fprintf(stderr,
					"verify: MISMATCH at batch entry %u (mock_fh=%d, nbytes=%lu)\n",
					i,
					(int)iter->data->entries[
						(iter->data->index - iter->opts->batch_size + i)
						% iter->data->n_entries].file,
					(unsigned long)nbytes);
				return EIO;
			}
		}
	}

	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		ds_file_handle_deregister(io->handles[i]);
	}
	return 0;
}

int
fil_opends_register_entry(struct fil_iter *iter, struct xal_inode *file_inode,
			  uint64_t *mock_fh_out)
{
	struct xal *xal = iter->devs[0]->xal;
	uint64_t lba_nbytes = xnvme_dev_get_geo(iter->devs[0]->dev)->lba_nbytes;
	uint32_t xal_blksize = xal_get_sb_blocksize(xal);
	struct homi_extent *ext_buf;
	uint32_t n_ext = file_inode->content.extents.count;
	uint64_t file_offset = 0;
	int rc;

	ext_buf = malloc(sizeof(*ext_buf) * n_ext);
	if (!ext_buf) {
		fprintf(stderr, "Could not allocate extent buffer for %s\n",
			file_inode->name);
		return ENOMEM;
	}

	for (uint32_t e = 0; e < n_ext; e++) {
		struct xal_extent xext = file_inode->content.extents.extent[e];
		ext_buf[e].file_offset = file_offset;
		ext_buf[e].slba = xal_fsbno_offset(xal, xext.start_block) / lba_nbytes;
		ext_buf[e].length = (uint64_t)xext.nblocks * xal_blksize;
		file_offset += ext_buf[e].length;
	}

	rc = fs_mock_register(ext_buf, n_ext, file_inode->size);
	free(ext_buf);
	if (rc < 0) {
		fprintf(stderr, "fs_mock_register(%s): %d\n", file_inode->name, -rc);
		return -rc;
	}
	*mock_fh_out = (uint64_t)rc;
	return 0;
}

int
fil_opends_io_alloc(struct fil_iter *iter)
{
	uint32_t batch_size = iter->opts->batch_size;
	int err;
	ds_file_error_t derr;

	iter->opends_io = calloc(1, sizeof(struct fil_opends_io));
	if (!iter->opends_io) {
		err = errno;
		fprintf(stderr, "Could not allocate OpenDS IO struct: %d\n", err);
		return err;
	}

	iter->opends_io->handles = malloc(sizeof(ds_file_handle_t) * batch_size);
	if (!iter->opends_io->handles) {
		err = errno;
		fprintf(stderr, "Could not allocate ds_file handles: %d\n", err);
		return err;
	}

	iter->opends_io->expected = malloc(sizeof(size_t) * batch_size);
	if (!iter->opends_io->expected) {
		err = errno;
		fprintf(stderr, "Could not allocate array of expected values: %d\n", err);
		return err;
	}

	iter->opends_io->actual = malloc(sizeof(ssize_t) * batch_size);
	if (!iter->opends_io->actual) {
		err = errno;
		fprintf(stderr, "Could not allocate array of actual values: %d\n", err);
		return err;
	}

	iter->opends_io->streams = malloc(sizeof(cudaStream_t) * batch_size);
	if (!iter->opends_io->streams) {
		err = errno;
		fprintf(stderr, "Could not allocate array of CUDA Streams: %d\n", err);
		return err;
	}

	for (uint32_t i = 0; i < batch_size; i++) {
		err = cudaStreamCreateWithFlags(&iter->opends_io->streams[i],
						cudaStreamNonBlocking);
		if (err) {
			fprintf(stderr, "Could not setup CUDA Stream, err: %d\n", err);
			return err;
		}
		derr = ds_file_stream_register(iter->opends_io->streams[i], 0);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_stream_register: %s\n",
				ds_file_op_status_error(derr.err));
			return derr.err;
		}
	}

	if (iter->opts->verify) {
		iter->opends_io->verify_buf_size = iter->buffer_size;
		err = cudaMalloc(&iter->opends_io->verify_dev_buf,
				 iter->buffer_size);
		if (err) {
			fprintf(stderr, "cudaMalloc(verify_dev_buf): %d\n", err);
			return err;
		}
		derr = ds_file_buf_register(iter->opends_io->verify_dev_buf,
					    iter->buffer_size, 0);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr,
				"ds_file_buf_register(verify_dev_buf): %s\n",
				ds_file_op_status_error(derr.err));
			return derr.err;
		}
		iter->opends_io->verify_host_async = malloc(iter->buffer_size);
		iter->opends_io->verify_host_sync = malloc(iter->buffer_size);
		if (!iter->opends_io->verify_host_async ||
		    !iter->opends_io->verify_host_sync) {
			fprintf(stderr, "Could not allocate verify host buffers\n");
			return ENOMEM;
		}
	}
	return 0;
}

void
fil_opends_io_free(struct fil_iter *iter)
{
	if (!iter->opends_io)
		return;
	free(iter->opends_io->handles);
	free(iter->opends_io->expected);
	free(iter->opends_io->actual);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		ds_file_stream_deregister(iter->opends_io->streams[i]);
		cudaStreamDestroy(iter->opends_io->streams[i]);
	}
	free(iter->opends_io->streams);
	if (iter->opends_io->verify_dev_buf) {
		ds_file_buf_deregister(iter->opends_io->verify_dev_buf);
		cudaFree(iter->opends_io->verify_dev_buf);
	}
	free(iter->opends_io->verify_host_async);
	free(iter->opends_io->verify_host_sync);
	free(iter->opends_io);
	iter->opends_io = NULL;
}
