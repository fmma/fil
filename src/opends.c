#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>

#include <ds_file.h>
#include <fs_mock.h>

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

		if (fs_mock_get_size((uint32_t)mock_fh, &nbytes) < 0) {
			fprintf(stderr, "fs_mock_get_size(%d) failed\n", mock_fh);
			return EIO;
		}

		buf_id = device->buf++ % device->n_buffers;
		buffer = device->buffers[buf_id];

		iter->output->buf_len[buf_id] = nbytes;
		iter->output->labels[buf_id] = 0;
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
