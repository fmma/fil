#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>

#include <ds_file.h>
#include <ds_file_async.h>

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
	struct fil_opends_rec *rec;
	ds_file_handle_t fh;
	ds_file_error_t derr;
	ssize_t got;
	int fd;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		rec = &iter->data->opends_recs[entry.file];
		nbytes = rec->size;

		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];

		iter->output->buf_len[buf_id] = nbytes;
		iter->output->labels[buf_id] = (uint32_t)entry.dir;
		iter->stats->bytes += nbytes;
		iter->stats->io++;

		fd = open(rec->path, O_RDONLY);
		if (fd == -1) {
			int e = errno;
			fprintf(stderr, "open(%s): %d\n", rec->path, e);
			return e;
		}

		derr = ds_file_handle_register(&fh, fd);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_handle_register(%s): %s\n",
				rec->path, ds_file_op_status_error(derr.err));
			close(fd);
			return derr.err;
		}

		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->prep_time += ELAPSED(start, end);

		clock_gettime(CLOCK_MONOTONIC_RAW, &start);
		got = ds_file_read(fh, buffer, nbytes, 0, 0);
		clock_gettime(CLOCK_MONOTONIC_RAW, &end);
		iter->stats->io_time += ELAPSED(start, end);

		ds_file_handle_deregister(fh);
		close(fd);

		if (got < 0) {
			fprintf(stderr, "ds_file_read(%s): %s\n", rec->path,
				ds_file_op_status_error((ds_file_op_error_t)(-got)));
			return EIO;
		}
		if ((uint64_t)got != nbytes) {
			fprintf(stderr,
				"ds_file_read(%s) short: expected %lu, got %ld\n",
				rec->path, nbytes, got);
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
	struct fil_opends_rec *rec;
	ds_file_error_t derr;
	off_t offset = 0;
	int err;

	clock_gettime(CLOCK_MONOTONIC_RAW, &start);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		uint64_t nbytes;
		entry = iter->data->entries[iter->data->index++ % iter->data->n_entries];
		rec = &iter->data->opends_recs[entry.file];
		nbytes = rec->size;

		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];

		iter->output->buf_len[buf_id] = nbytes;
		iter->output->labels[buf_id] = (uint32_t)entry.dir;
		iter->stats->bytes += nbytes;
		iter->stats->io++;

		io->fds[i] = open(rec->path, O_RDONLY);
		if (io->fds[i] == -1) {
			err = errno;
			fprintf(stderr, "open(%s): %d\n", rec->path, err);
			return err;
		}

		derr = ds_file_handle_register(&io->handles[i], io->fds[i]);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_handle_register(%s): %s\n",
				rec->path, ds_file_op_status_error(derr.err));
			close(io->fds[i]);
			return derr.err;
		}

		io->expected[i] = nbytes;
		io->actual[i] = 0;

		derr = ds_file_read_async(io->handles[i], buffer,
					  &io->expected[i], &offset, &offset,
					  &io->actual[i], io->streams[i]);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_read_async(%s): %s\n",
				rec->path, ds_file_op_status_error(derr.err));
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

	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		ds_file_handle_deregister(io->handles[i]);
		close(io->fds[i]);
	}
	return 0;
}

int
fil_opends_io_alloc(struct fil_iter *iter)
{
	uint32_t batch_size = iter->opts->batch_size;
	int err;

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

	iter->opends_io->fds = malloc(sizeof(int) * batch_size);
	if (!iter->opends_io->fds) {
		err = errno;
		fprintf(stderr, "Could not allocate fd array: %d\n", err);
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
		ds_file_error_t derr =
			ds_file_stream_register(iter->opends_io->streams[i], 0);
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_stream_register: %s\n",
				ds_file_op_status_error(derr.err));
			return derr.err;
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
	free(iter->opends_io->fds);
	free(iter->opends_io->expected);
	free(iter->opends_io->actual);
	for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
		ds_file_stream_deregister(iter->opends_io->streams[i]);
		cudaStreamDestroy(iter->opends_io->streams[i]);
	}
	free(iter->opends_io->streams);
	free(iter->opends_io);
	iter->opends_io = NULL;
}
