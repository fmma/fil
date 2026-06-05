#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>

#include <libfil.h>
#include <fil_io.h>
#include <fil_iter.h>
#include <fil_util.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>
#include <ds_file.h>
#include <ds_file_async.h>
#include <libxal.h>
#include <libxnvme.h>

// Arbitrary value out of range of normal err
#define DATA_DIR_FOUND 9000

static int
inode_cmp(const void *a, const void *b)
{
	const struct xal_inode *inode_a = (const struct xal_inode *)a;
	const char *name_a = inode_a->name;
	const struct xal_inode *inode_b = (const struct xal_inode *)b;
	const char *name_b = inode_b->name;

	return strcmp(name_a, name_b);
}

static int
_xnvme_setup(struct fil_iter *iter, struct fil_dev *device, const char *uri)
{
	const char *backend = iter->opts->backend;
	struct xnvme_opts opts = xnvme_opts_default();
	struct xnvme_dev *dev;
	int err;
	CUfileError_t status;

	if (strcmp(backend, "io_uring") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 0;
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "io_uring_direct") == 0) {
		opts.be = "linux";
		opts.async = "io_uring";
		opts.direct = 1;
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "spdk") == 0) {
		opts.be = "spdk";
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "aisio-cpu") == 0) {
		opts.be = "upcie";
		iter->type = FIL_CPU;
	} else if (strcmp(backend, "aisio-gpu") == 0) {
		iter->type = FIL_GPU;
	} else if (strcmp(backend, "posix") == 0) {
		opts.be = "linux";
		iter->type = FIL_FILE;
	} else if (strcmp(backend, "gds") == 0) {
		opts.be = "linux";
		iter->type = FIL_FILE;
		status = cuFileDriverOpen();
		if (status.err != CU_FILE_SUCCESS) {
			fprintf(stderr, "Could not open cuFile driver: %d\n", status.err);
			return status.err;
		}
	} else if (strcmp(backend, "opends") == 0) {
		opts.be = "upcie";
		iter->type = FIL_OPENDS;
	} else {
		fprintf(stderr, "Invalid backend: %s\n", backend);
		return EINVAL;
	}

	if (iter->type == FIL_GPU) {
		return -ENOSYS;
	}

	dev = xnvme_dev_open(uri, &opts);
	if (!dev) {
		err = errno;
		fprintf(stderr, "xnvme_dev_open(): %d\n", err);
		return err;
	}

	err = xnvme_dev_derive_geo(dev);
	if (err) {
		xnvme_dev_close(dev);
		fprintf(stderr, "xnvme_dev_derive_geo(): %d\n", err);
		return err;
	}

	if (iter->type == FIL_CPU) {
		err = xnvme_queue_init(dev, iter->opts->queue_depth, 0, &device->queue);
		if (err) {
			xnvme_dev_close(dev);
			fprintf(stderr, "xnvme_queue_init(): %d\n", err);
			return err;
		}
	}

	device->dev = dev;
	return 0;
}

static int
find_buffer_size(struct xal *FIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
		 int FIL_UNUSED(level))
{
	struct fil_stats *stats = (struct fil_stats *)cb_args;

	if (xal_inode_is_file(inode)) {
		stats->n_files++;
		stats->avg_file_size += inode->size;
		if (inode->size > stats->max_file_size) {
			stats->max_file_size = inode->size;
		}
	}
	return 0;
}

static int
find_data_dir(struct xal *FIL_UNUSED(xal), struct xal_inode *inode, void *cb_args,
	      int FIL_UNUSED(level))
{
	struct fil_dev *dev = (struct fil_dev *)cb_args;

	if (strcmp(dev->data_dir, inode->name) == 0) {
		dev->root_inode = inode;
		return DATA_DIR_FOUND; // break
	}

	return 0; // continue
}

static void
path_prepend(struct xal *xal, char *path, struct xal_inode *node)
{
	if (node->name[0] == '\0' || node->parent_idx == XAL_POOL_IDX_NONE) {
		return;
	}
	path_prepend(xal, path, xal_inode_at(xal, node->parent_idx));
	strcat(path, "/");
	strcat(path, node->name);
}

static void
_find_prefix(struct fil_iter *iter)
{
	char *prefix;
	struct fil_dev *device;
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		device = iter->devs[i];
		prefix = device->file_io->prefix;
		strcpy(prefix, iter->opts->mnt);
		path_prepend(device->xal, prefix, device->root_inode);
	}
}

static int
_xal_setup(struct fil_iter *iter, struct fil_dev *device)
{
	struct xal *xal;
	struct xal_opts xal_opts = {0};
	struct xal_inode *root;
	uint32_t xal_blksize;
	int err;

	xal_opts.be = XAL_BACKEND_XFS;

	err = xal_open(device->dev, &xal, &xal_opts);
	if (err) {
		fprintf(stderr, "xal_open(): %d\n", err);
		return err;
	}

	err = xal_dinodes_retrieve(xal);
	if (err) {
		fprintf(stderr, "xal_dinodes_retrieve(): %d\n", err);
		xal_close(xal);
		return err;
	}

	err = xal_index(xal);
	if (err) {
		fprintf(stderr, "xal_index(): %d\n", err);
		xal_close(xal);
		return err;
	}

	device->data_dir = iter->opts->data_dir;

	root = xal_get_root(xal);
	err = xal_walk(xal, root, find_data_dir, device);
	switch (err) {
	case DATA_DIR_FOUND:
		break;

	case 0: // Root dir not found
		fprintf(stderr, "Couldn't find root directory: %s\n", device->data_dir);
		xal_close(xal);
		return ENOENT;

	default:
		fprintf(stderr, "xal_walk(find_data_dir): %d\n", err);
		xal_close(xal);
		return err;
	}

	if (!iter->buffer_size) {
		err = xal_walk(xal, device->root_inode, find_buffer_size, iter->stats);
		if (err) {
			fprintf(stderr, "xal_walk(find_buffer_size): %d\n", err);
			return err;
		}

		xal_blksize = xal_get_sb_blocksize(xal);

		iter->stats->avg_file_size = iter->stats->avg_file_size / iter->stats->n_files;
		// Align to page size
		iter->buffer_size = (1 + ((iter->stats->max_file_size - 1) / xal_blksize)) *
				    (xal_blksize);
	}

	// Sort the directories so we can derive labels
	if (device->root_inode->content.dentries.count > 1) {
		qsort(xal_inode_at(xal, device->root_inode->content.dentries.inodes_idx),
		      device->root_inode->content.dentries.count, sizeof(struct xal_inode),
		      inode_cmp);
	}
	device->xal = xal;
	return 0;
}

static int
_create_entries(struct fil_iter *iter)
{
	struct fil_entry *entries;
	struct xal *xal = iter->devs[0]->xal;
	struct xal_dentries root_dentries;
	uint64_t n_entries = 0;
	int err;
	int k;

	root_dentries = iter->devs[0]->root_inode->content.dentries;

	for (uint32_t i = 0; i < root_dentries.count; i++) {
		n_entries += xal_inode_at(xal, root_dentries.inodes_idx + i)
				 ->content.dentries.count;
	}

	entries = malloc(sizeof(struct fil_entry) * n_entries);
	if (!entries) {
		err = errno;
		fprintf(stderr, "Could not allocate entries: %d\n", err);
		return err;
	}

	k = 0;
	for (uint32_t i = 0; i < root_dentries.count; i++) {
		struct xal_inode *dir_inode = xal_inode_at(xal, root_dentries.inodes_idx + i);
		for (uint32_t j = 0; j < dir_inode->content.dentries.count; j++) {
			entries[k].dir = i;
			entries[k].file = j;
			k++;
		}
	}

	iter->data->entries = entries;
	iter->data->n_entries = n_entries;

	return 0;
}

static int
_label_cmp(const void *a, const void *b)
{
	return strcmp((const char *)a, (const char *)b);
}

/* opends backend: enumerate the dataset by walking the mount, following the
 * mnt/<data_dir>/<label-dir>/<file> layout and recording each file's path and
 * size. entry.dir is the (sorted) label index and entry.file indexes
 * opends_recs. */
static int
_homi_create_entries(struct fil_iter *iter)
{
	const char *mnt = iter->opts->mnt;
	const char *data_dir = iter->opts->data_dir;
	char base[PATH_MAX];
	char (*labels)[NAME_MAX + 1] = NULL;
	uint32_t n_labels = 0, cap_labels = 0;
	struct fil_opends_rec *recs = NULL;
	struct fil_entry *entries = NULL;
	uint64_t n = 0, cap = 0, max_size = 0, total_size = 0;
	struct dirent *de;
	int err;
	DIR *bd;

	snprintf(base, sizeof(base), "%s/%s", mnt, data_dir);

	bd = opendir(base);
	if (!bd) {
		err = errno;
		fprintf(stderr, "opendir(%s): %d\n", base, err);
		return err;
	}
	while ((de = readdir(bd))) {
		char sub[PATH_MAX];
		struct stat st;

		if (de->d_name[0] == '.')
			continue;
		snprintf(sub, sizeof(sub), "%s/%s", base, de->d_name);
		if (stat(sub, &st) != 0 || !S_ISDIR(st.st_mode))
			continue;
		if (n_labels == cap_labels) {
			cap_labels = cap_labels ? cap_labels * 2 : 16;
			void *p = realloc(labels, sizeof(*labels) * cap_labels);
			if (!p) {
				free(labels);
				closedir(bd);
				return ENOMEM;
			}
			labels = p;
		}
		snprintf(labels[n_labels], NAME_MAX + 1, "%s", de->d_name);
		n_labels++;
	}
	closedir(bd);

	if (n_labels > 1)
		qsort(labels, n_labels, sizeof(*labels), _label_cmp);

	for (uint32_t li = 0; li < n_labels; li++) {
		char dirpath[PATH_MAX];
		DIR *dd;

		snprintf(dirpath, sizeof(dirpath), "%s/%s", base, labels[li]);
		dd = opendir(dirpath);
		if (!dd) {
			err = errno;
			fprintf(stderr, "opendir(%s): %d\n", dirpath, err);
			free(labels);
			free(recs);
			free(entries);
			return err;
		}
		while ((de = readdir(dd))) {
			char fpath[PATH_MAX];
			struct stat st;

			if (de->d_name[0] == '.')
				continue;
			snprintf(fpath, sizeof(fpath), "%s/%s", dirpath, de->d_name);
			if (stat(fpath, &st) != 0 || !S_ISREG(st.st_mode))
				continue;
			if (n == cap) {
				cap = cap ? cap * 2 : 256;
				void *pr = realloc(recs, sizeof(*recs) * cap);
				void *pe = realloc(entries, sizeof(*entries) * cap);
				if (!pr || !pe) {
					free(pr ? pr : recs);
					free(pe ? pe : entries);
					free(labels);
					closedir(dd);
					return ENOMEM;
				}
				recs = pr;
				entries = pe;
			}
			snprintf(recs[n].path, sizeof(recs[n].path), "%s", fpath);
			recs[n].size = (uint64_t)st.st_size;
			entries[n].dir = li;
			entries[n].file = n;
			if (recs[n].size > max_size)
				max_size = recs[n].size;
			total_size += recs[n].size;
			n++;
		}
		closedir(dd);
	}
	free(labels);

	if (n == 0) {
		fprintf(stderr, "No files found under %s\n", base);
		free(recs);
		free(entries);
		return ENOENT;
	}

	iter->data->entries = entries;
	iter->data->n_entries = n;
	iter->data->opends_recs = recs;
	iter->data->n_recs = n;

	iter->stats->n_files = n;
	iter->stats->max_file_size = max_size;
	iter->stats->avg_file_size = total_size / n;

	if (!iter->buffer_size) {
		uint64_t pg = 4096;
		iter->buffer_size = (1 + ((max_size - 1) / pg)) * pg;
	}

	return 0;
}

static int
_opends_cuda_ctx(void)
{
	static CUcontext ctx;
	CUdevice cudev;
	CUresult cr;

	cr = cuInit(0);
	if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "cuInit: %d\n", cr);
		return EIO;
	}
	cr = cuDeviceGet(&cudev, 0);
	if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "cuDeviceGet: %d\n", cr);
		return EIO;
	}
	cr = cuCtxCreate(&ctx, 0, cudev);
	if (cr != CUDA_SUCCESS) {
		fprintf(stderr, "cuCtxCreate: %d\n", cr);
		return EIO;
	}
	return 0;
}

static int
_alloc(struct fil_iter *iter, uint32_t n_buffers)
{
	int err;
	iter->output = malloc(sizeof(struct fil_output));
	if (!iter->output) {
		err = errno;
		fprintf(stderr, "Could not allocate output struct: %d\n", err);
	}
	iter->output->n_buffers = n_buffers;

	iter->output->buffers = malloc(sizeof(void *) * iter->output->n_buffers);
	if (!iter->output->buffers) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
		return err;
	}

	iter->output->labels = malloc(sizeof(uint32_t) * iter->output->n_buffers);
	if (!iter->output->labels) {
		err = errno;
		fprintf(stderr, "Could not allocate array of labels: %d\n", err);
		return err;
	}

	iter->output->buf_len = malloc(sizeof(uint64_t) * iter->output->n_buffers);
	if (!iter->output->buf_len) {
		err = errno;
		fprintf(stderr, "Could not allocate array of buffer lengths: %d\n", err);
		return err;
	}
	memset(iter->output->buf_len, 0, sizeof(uint64_t) * iter->output->n_buffers);

	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];
		device->n_buffers = iter->output->n_buffers / iter->n_devs;
		device->buf = 0;
		device->buffers = malloc(sizeof(void *) * device->n_buffers);
		if (!device->buffers) {
			err = errno;
			fprintf(stderr, "Could not allocate array of buffers: %d\n", err);
			return err;
		}
		for (uint32_t j = 0; j < device->n_buffers; j++) {
			switch (iter->type) {
			case FIL_GPU:
				return -ENOSYS;
			case FIL_CPU:
				device->buffers[j] =
				    xnvme_buf_alloc(device->dev, iter->buffer_size);
				break;
			case FIL_FILE:
				err = cudaMalloc(&device->buffers[j], iter->buffer_size);
				if (err) {
					fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i,
						err);
					return err;
				}
				break;
			case FIL_OPENDS: {
				ds_file_error_t derr;
				err = cudaMalloc(&device->buffers[j], iter->buffer_size);
				if (err) {
					fprintf(stderr, "cudaMalloc(buffers[%d]): %d\n", i, err);
					return err;
				}
				derr = ds_file_buf_register(device->buffers[j],
							    iter->buffer_size, 0);
				if (derr.err != DS_FILE_SUCCESS) {
					fprintf(stderr,
						"ds_file_buf_register(buffers[%d]): %s\n",
						i, ds_file_op_status_error(derr.err));
					return derr.err;
				}
				break;
			}
			}
			if (!device->buffers[j]) {
				err = errno;
				fprintf(stderr, "Could not allocate buffers[%d]: %d\n", i, err);
				return err;
			}
			iter->output->buffers[j + i * device->n_buffers] = device->buffers[j];
		}

		if (iter->type == FIL_CPU) {
			device->cpu_io = malloc(sizeof(struct fil_cpu_io));
			if (!device->cpu_io) {
				err = errno;
				fprintf(stderr, "Could not allocate IO struct: %d\n", err);
				return err;
			}

			device->cpu_io->slbas = malloc(sizeof(uint64_t) * device->n_buffers);
			if (!device->cpu_io->slbas) {
				err = errno;
				fprintf(stderr, "Could not allocate array for slbas: %d\n", err);
				return err;
			}

			device->cpu_io->elbas = malloc(sizeof(uint64_t) * device->n_buffers);
			if (!device->cpu_io->elbas) {
				err = errno;
				fprintf(stderr, "Could not allocate array for elbas: %d\n", err);
				return err;
			}
		} else if (iter->type == FIL_FILE) {
			device->file_io = malloc(sizeof(struct fil_file_io));
			if (!device->file_io) {
				err = errno;
				fprintf(stderr, "Could not allocate IO struct: %d\n", err);
				return err;
			}
			device->file_io->buffer = malloc(iter->buffer_size);
			if (!device->file_io->buffer) {
				err = errno;
				fprintf(stderr, "Could not allocate bounce buffer: %d\n", err);
				return err;
			}
		}
	}

	if (iter->opts->async && iter->type == FIL_FILE) {
		iter->gds_io = malloc(sizeof(struct fil_gds_io));
		if (!iter->gds_io) {
			err = errno;
			fprintf(stderr, "Could not allocate GDS IO struct: %d\n", err);
			return err;
		}

		iter->gds_io->descr = malloc(sizeof(CUfileDescr_t) * iter->opts->batch_size);
		if (!iter->gds_io->descr) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile descriptors: %d\n", err);
			return err;
		}

		iter->gds_io->handle = malloc(sizeof(CUfileHandle_t) * iter->opts->batch_size);
		if (!iter->gds_io->handle) {
			err = errno;
			fprintf(stderr, "Could not allocate cuFile handles: %d\n", err);
			return err;
		}

		iter->gds_io->expected = malloc(sizeof(size_t) * iter->opts->batch_size);
		if (!iter->gds_io->expected) {
			err = errno;
			fprintf(stderr, "Could not allocate array of expected values: %d\n", err);
			return err;
		}

		iter->gds_io->actual = malloc(sizeof(ssize_t) * iter->opts->batch_size);
		if (!iter->gds_io->actual) {
			err = errno;
			fprintf(stderr, "Could not allocate array of actual values: %d\n", err);
			return err;
		}

		iter->gds_io->streams = malloc(sizeof(cudaStream_t) * iter->opts->batch_size);
		if (!iter->gds_io->streams) {
			err = errno;
			fprintf(stderr, "Could not allocate array of CUDA Streams: %d\n", err);
			return err;
		}

		for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
			err = cudaStreamCreateWithFlags(&iter->gds_io->streams[i],
							cudaStreamNonBlocking);
			if (err) {
				fprintf(stderr, "Could not setup CUDA Stream, err: %d\n", err);
				return err;
			}
		}
	}

	if (iter->opts->async && iter->type == FIL_OPENDS) {
		err = fil_opends_io_alloc(iter);
		if (err) {
			return err;
		}
	}

	return 0;
}

void
fil_term(struct fil_iter *iter)
{
	for (uint32_t i = 0; i < iter->n_devs; i++) {
		struct fil_dev *device = iter->devs[i];
		switch (iter->type) {
		case FIL_GPU:
			break;
		case FIL_CPU:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				xnvme_buf_free(device->dev, device->buffers[j]);
			}
			xnvme_queue_term(device->queue);
			break;
		case FIL_FILE:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				cudaFree(device->buffers[j]);
			}
			break;
		case FIL_OPENDS:
			for (uint32_t j = 0; j < device->n_buffers; j++) {
				ds_file_buf_deregister(device->buffers[j]);
				cudaFree(device->buffers[j]);
			}
			break;
		}
		if (device->xal) {
			xal_close(device->xal);
		}
		if (device->dev) {
			xnvme_dev_close(device->dev);
		}
		if (iter->type != FIL_OPENDS) {
			cuFileDriverClose();
		}
		if (device->cpu_io) {
			free(device->cpu_io->slbas);
			free(device->cpu_io->elbas);
			free(device->cpu_io);
		} else if (device->file_io) {
			free(device->file_io->buffer);
			free(device->file_io);
		}
		free(device->buffers);
		free(device);
	}
	fil_opends_io_free(iter);
	if (iter->type == FIL_OPENDS) {
		ds_file_driver_close();
	}
	if (iter->gds_io) {
		free(iter->gds_io->descr);
		free(iter->gds_io->handle);
		free(iter->gds_io->expected);
		free(iter->gds_io->actual);
		for (uint32_t i = 0; i < iter->opts->batch_size; i++) {
			cudaStreamDestroy(iter->gds_io->streams[i]);
		}
		free(iter->gds_io->streams);
	}
	if (iter->data) {
		free(iter->data->entries);
		free(iter->data->opends_recs);
		free(iter->data);
	}
	free(iter->opts);
	if (iter->output) {
		free(iter->output->buffers);
		free(iter->output->labels);
		free(iter->output->buf_len);
		free(iter->output);
	}
	free(iter->stats);
	free(iter);
}

static int
_init_stats(struct fil_stats **stats)
{
	int err;
	struct fil_stats *_stats = malloc(sizeof(struct fil_stats));
	if (!_stats) {
		err = errno;
		fprintf(stderr, "Could not allocate stats: %d\n", err);
		return err;
	}
	memset(_stats, 0, sizeof(struct fil_stats));
	*stats = _stats;
	return 0;
}

int
fil_init(struct fil_iter **iter, char **dev_uris, uint32_t n_devs, struct fil_opts *opts)
{
	struct fil_iter *_iter;
	srand(time(NULL));
	int err;

	if (opts->batch_size % n_devs != 0) {
		fprintf(stderr, "Batch size (%u) not divisible by number of devices (%u)\n",
			opts->batch_size, n_devs);
	}

	if (opts->buffered && strcmp(opts->backend, "posix") != 0) {
		fprintf(stderr, "opts->buffered == true is only compatible with POSIX backend");
		return EINVAL;
	}

	if (opts->async && strcmp(opts->backend, "gds") != 0
			&& strcmp(opts->backend, "opends") != 0) {
		fprintf(stderr,
			"opts->async is only compatible with gds or opends backends\n");
		return EINVAL;
	}

	if (strcmp(opts->backend, "opends") == 0 && n_devs != 1) {
		fprintf(stderr, "opends backend supports a single device only (got %u)\n",
			n_devs);
		return EINVAL;
	}

	_iter = malloc(sizeof(struct fil_iter));
	if (!_iter) {
		err = errno;
		fprintf(stderr, "Could not allocate iter: %d\n", err);
		return err;
	}
	memset(_iter, 0, sizeof(struct fil_iter));

	_iter->opts = malloc(sizeof(struct fil_opts));
	if (!_iter->opts) {
		err = errno;
		fprintf(stderr, "Conld not allocate opts: %d\n", err);
		return err;
	}
	memcpy(_iter->opts, opts, sizeof(*opts));

	_iter->devs = malloc(sizeof(struct fil_dev *) * n_devs);
	if (!_iter->devs) {
		err = errno;
		fprintf(stderr, "Could not allocate devices: %d\n", err);
		fil_term(_iter);
		return err;
	}

	err = _init_stats(&_iter->stats);
	if (err) {
		fil_term(_iter);
		return err;
	}

	if (opts->data_dir[0] == '\0') {
		fprintf(stderr, "data_dir is required\n");
		fil_term(_iter);
		return EINVAL;
	}

	for (uint32_t i = 0; i < n_devs; i++) {
		struct fil_dev *device = malloc(sizeof(struct fil_dev));
		if (!device) {
			err = errno;
			fprintf(stderr, "Could not allocate handle for %s: %d\n", dev_uris[i], err);
			fil_term(_iter);
			return err;
		}
		memset(device, 0, sizeof(struct fil_dev));

		if (strcmp(opts->backend, "opends") == 0) {
			/* The opends backend reads through the OpenDS file API, so
			 * enumerate the dataset by readdir on the mount rather than
			 * opening the device or walking xal. */
			_iter->type = FIL_OPENDS;
			device->data_dir = opts->data_dir;
		} else {
			err = _xnvme_setup(_iter, device, dev_uris[i]);
			if (err) {
				fprintf(stderr, "xNVMe setup failed for %s: %d\n", dev_uris[i],
					err);
				fil_term(_iter);
				return err;
			}
			err = _xal_setup(_iter, device);
			if (err) {
				fprintf(stderr, "XAL setup failed for %s: %d\n", dev_uris[i],
					err);
				xnvme_dev_close(device->dev);
				fil_term(_iter);
				return err;
			}
		}
		_iter->devs[i] = device;
		_iter->n_devs++;
	}

	_iter->data = calloc(1, sizeof(struct fil_data));
	if (!_iter->data) {
		err = errno;
		fprintf(stderr, "Could not allocate data: %d\n", err);
		fil_term(_iter);
		return err;
	}

	if (_iter->type == FIL_OPENDS) {
		/* The upcie-cuda backend DMAs into GPU memory, so a driver-API
		 * CUDA context must be current before ds_file_driver_open. */
		err = _opends_cuda_ctx();
		if (err) {
			fil_term(_iter);
			return err;
		}
		ds_file_error_t derr = ds_file_driver_open();
		if (derr.err != DS_FILE_SUCCESS) {
			fprintf(stderr, "ds_file_driver_open: %s\n",
				ds_file_op_status_error(derr.err));
			fil_term(_iter);
			return derr.err;
		}
	}

	// Create an entry for every file in every directory.
	if (_iter->type == FIL_OPENDS) {
		err = _homi_create_entries(_iter);
	} else {
		err = _create_entries(_iter);
	}
	if (err) {
		fil_term(_iter);
		return err;
	}

	err = _alloc(_iter, _iter->opts->batch_size);
	if (err) {
		fil_term(_iter);
		return err;
	}
	switch (_iter->type) {
	case FIL_GPU:
		_iter->io_fn = fil_gpu_submit;
		break;
	case FIL_CPU:
		_iter->io_fn = fil_cpu_submit;
		break;
	case FIL_FILE:
		if (_iter->opts->async) {
			_iter->io_fn = fil_gds_async_submit;
		} else {
			_iter->io_fn = fil_file_submit;
		}
		_find_prefix(_iter);
		break;
	case FIL_OPENDS:
		if (_iter->opts->async) {
			_iter->io_fn = fil_opends_async_submit;
		} else {
			_iter->io_fn = fil_opends_submit;
		}
		break;
	}

	FIL_SHUFFLE(_iter->data->entries, struct fil_entry, _iter->data->n_entries,
		    uint64_t);

	(*iter) = _iter;

	return 0;
}

int
fil_next(struct fil_iter *iter, struct fil_output **output)
{
	int err;

	if (iter->data->entries && iter->data->index >= iter->data->n_entries) {
		iter->data->index = 0;
		FIL_SHUFFLE(iter->data->entries, struct fil_entry, iter->data->n_entries, uint64_t);
	}

	err = iter->io_fn(iter);
	if (err) {
		fprintf(stderr, "Data reading failed, err: %d\n", err);
		return err;
	}

	*output = iter->output;

	return 0;
}

struct fil_opts
fil_opts_default()
{
	struct fil_opts opts = {.data_dir = "",
				.mnt = "/mnt",
				.backend = "aisio-cpu",
				.iosize = 4096,
				.gpu_nqueues = 128,
				.gpu_tbsize = 64,
				.queue_depth = 1024,
				.batch_size = 1,
				.buffered = false,
				.async = false};

	return opts;
}

struct fil_stats *
fil_get_stats(struct fil_iter *iter)
{
	return iter->stats;
}
