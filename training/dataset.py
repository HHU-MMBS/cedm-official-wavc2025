# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None


class Dataset(torch.utils.data.Dataset):
    """Abstract base class for datasets."""
    def __init__(self,
        name,                         # Name of the dataset.
        raw_shape,                    # Shape of the raw im,age data (NCHW).
        max_size          = 0,        # Artificially limit the size of the dataset. 0 = no limit. Applied before xflip.
        use_labels        = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip             = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed       = 0,        # Random seed to use when applying max_size.
        cache             = False,    # Cache images in CPU memory.
        subclass          = -1,       # Restrict the dataset to only one class/cluster. -1 = disabled.
        subset_idx        = None,     # Slice indices that specify a subset to restrict the dataset to.
        shuffle_subset    = True,     # Shuffle the subset. Only works if dataset size has been restricted.
        pseudo_label_path = None,     # Path to load pseudo-labels from.
        p_uncond          = 0.0,      # Probability to drop the condition for training for classifier free guidance.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._check_pseudo_labels(pseudo_label_path)
        self._cache = cache
        self._cached_images = dict()  # {raw_idx: np.ndarray, ...}
        self._label_shape = None
        self._max_size = max_size
        self.p_uncond = p_uncond

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        assert not (subset_idx is not None and subclass > -1), "Cannot use subset_idx and subclass at the same time"
        if subclass > -1:  # Select subclass
            sub_idx = self._load_raw_labels() == subclass
            self._raw_idx = self._raw_idx[sub_idx]
        if subset_idx is not None:
            self._raw_idx = self._raw_idx[subset_idx]
        if (max_size != 0) and (self._raw_idx.size > max_size):
            if shuffle_subset:
                np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])
        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    @staticmethod
    def _read_pseudo_labels(path):
        unique_ids, inverse_indices = torch.unique(torch.load(path), return_inverse=True)
        mapped_ids = torch.arange(len(unique_ids))
        tensor_mapped = mapped_ids[inverse_indices]
        return tensor_mapped

    def _check_pseudo_labels(self, pseudo_label_path):
        if pseudo_label_path is not None:
            assert self._use_labels
            assert os.path.isfile(pseudo_label_path) and pseudo_label_path.endswith('.pt'), "pseudo label path doesn't exist or it is not a .pt file"
            self._use_pseudo_labels = True
            self._psuedo_label_path = pseudo_label_path
            self._raw_labels = self._read_pseudo_labels(self._psuedo_label_path)
        else:
            self._use_pseudo_labels = False

    def _get_raw_labels(self):
        if self._raw_labels is None:
            if self._use_pseudo_labels:
                self._raw_labels = self._read_pseudo_labels(self._psuedo_label_path)
            else:
                # Default behavior: load labels from dataset.json.
                self._raw_labels = self._load_raw_labels() if self._use_labels else None
                if self._raw_labels is None:
                    self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
                assert isinstance(self._raw_labels, np.ndarray)
                assert self._raw_labels.shape[0] == self._raw_shape[0]
                assert self._raw_labels.dtype in [np.float32, np.int64]
                if self._raw_labels.dtype == np.int64:
                    assert self._raw_labels.ndim == 1
                    assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        # Pseudo-labels
        if self._use_pseudo_labels:
            label = self._get_raw_labels()[idx]
            if np.random.rand() > self.p_uncond:
                return torch.nn.functional.one_hot(label, num_classes=self.label_dim).long()
            else:
                return torch.zeros(self.label_dim, dtype=torch.long)
        # GT labels
        label = self._get_raw_labels()[self._raw_idx[idx]].astype(np.int64)
        onehot = np.zeros(self.label_shape, dtype=np.float32)
        onehot[label] = 1 if np.random.rand() > self.p_uncond else 0
        return onehot.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def raw_idx(self):
        return self._raw_idx

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        """number of classes/clusters"""
        if self._use_pseudo_labels:
            self._label_shape = [int(torch.max(self._raw_labels)) + 1]
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class ImageFolderDataset(Dataset):
    """Dataset subclass that loads images recursively from the specified directoryor ZIP file."""
    def __init__(self,
                 path,                     # Path to directory or zip.
                 resolution       = None,  # Ensure specific resolution, None = highest available.
                 use_pyspng       = True,  # Use pyspng if available.
                 pseudo_label_path= None,  # Path to pseudo labels.
                 **super_kwargs,           # Additional arguments for the Dataset base class.
                 ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None
        self._use_pseudo_labels = False if pseudo_label_path is None else True

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in
                                os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        self.raw_shape = raw_shape
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, pseudo_label_path=pseudo_label_path, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image

    def _load_raw_labels(self):
        if self._use_pseudo_labels:
            labels = super()._raw_labels
            print(labels.shape, len(labels))
        else:
            fname = 'dataset.json'
            if fname not in self._all_fnames:
                return None
            with self._open_file(fname) as f:
                labels = json.load(f)['labels']
            if labels is None:
                return None
            labels = dict(labels)
            labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
            labels = np.array(labels)
            labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
