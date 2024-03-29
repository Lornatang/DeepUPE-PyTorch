# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import queue
import threading

import cv2
import numpy as np
import torch
from natsort import natsorted
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from imgproc import image_to_tensor

__all__ = [
    "ImageDataset", "PairedImageDataset",
    "PrefetchGenerator", "PrefetchDataLoader", "CPUPrefetcher", "CUDAPrefetcher",
]


class ImageDataset(Dataset):
    """Define training/valid dataset loading methods.

    Args:
        gt_images_dir (str): Train/Valid dataset address.
        input_images_dir (str): Expert dataset address.
        low_resolution_size (int): Low resolution size of the training/valid dataset.
        high_resolution_size (int): High resolution size of the training/valid dataset.
    """

    def __init__(
            self,
            gt_images_dir: str,
            input_images_dir: str,
            low_resolution_size: int,
            high_resolution_size: int,
    ) -> None:
        super(ImageDataset, self).__init__()
        self.gt_image_files_name = natsorted(
            [os.path.join(gt_images_dir, image_file_name) for image_file_name in os.listdir(gt_images_dir)])
        self.input_image_files_name = natsorted(
            [os.path.join(input_images_dir, image_file_name) for image_file_name in os.listdir(input_images_dir)])
        self.low_resolution_size = low_resolution_size
        self.high_resolution_size = high_resolution_size

    def __getitem__(self, batch_index: int) -> [dict[str, Tensor], dict[str, Tensor]]:
        # Read a batch of image data
        gt_image = cv2.imread(self.gt_image_files_name[batch_index]).astype(np.float32) / 255.
        input_image = cv2.imread(self.input_image_files_name[batch_index]).astype(np.float32) / 255.

        # Image processing operations
        gt_image = cv2.resize(gt_image, (self.high_resolution_size, self.high_resolution_size))
        input_image = cv2.resize(input_image, (self.high_resolution_size, self.high_resolution_size))

        # BGR convert RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = image_to_tensor(gt_image, False, False)
        input_tensor = image_to_tensor(input_image, False, False)

        return {"gt": gt_tensor, "input": input_tensor}

    def __len__(self) -> int:
        return len(self.gt_image_files_name)


class PairedImageDataset(Dataset):
    """Define Test dataset loading methods.

    Args:
        gt_images_dir (str): ground truth image in test image
        input_images_dir (str): low-resolution image in test image
    """

    def __init__(self, gt_images_dir: str, input_images_dir: str) -> None:
        super(PairedImageDataset, self).__init__()
        # Get all image file names in folder
        self.gt_image_files_name = natsorted(
            [os.path.join(gt_images_dir, image_file_name) for image_file_name in os.listdir(gt_images_dir)])
        self.input_image_files_name = natsorted(
            [os.path.join(input_images_dir, image_file_name) for image_file_name in os.listdir(input_images_dir)])

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Read a batch of image data
        gt_image = cv2.imread(self.gt_image_files_name[batch_index]).astype(np.float32) / 255.
        input_image = cv2.imread(self.input_image_files_name[batch_index]).astype(np.float32) / 255.

        # BGR convert RGB
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        # Convert image data into Tensor stream format (PyTorch).
        # Note: The range of input and output is between [0, 1]
        gt_tensor = image_to_tensor(gt_image, False, False)
        input_tensor = image_to_tensor(input_image, False, False)

        return {"gt": gt_tensor,
                "input": input_tensor,
                "image_name": self.gt_image_files_name[batch_index]}

    def __len__(self) -> int:
        return len(self.gt_image_files_name)


class PrefetchGenerator(threading.Thread):
    """A fast data prefetch generator.

    Args:
        generator: Data generator.
        num_data_prefetch_queue (int): How many early data load queues.
    """

    def __init__(self, generator, num_data_prefetch_queue: int) -> None:
        threading.Thread.__init__(self)
        self.queue = queue.Queue(num_data_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self):
        return self


class PrefetchDataLoader(DataLoader):
    """A fast data prefetch dataloader.

    Args:
        num_data_prefetch_queue (int): How many early data load queues.
        kwargs (dict): Other extended parameters.
    """

    def __init__(self, num_data_prefetch_queue: int, **kwargs) -> None:
        self.num_data_prefetch_queue = num_data_prefetch_queue
        super(PrefetchDataLoader, self).__init__(**kwargs)

    def __iter__(self):
        return PrefetchGenerator(super().__iter__(), self.num_data_prefetch_queue)


class CPUPrefetcher:
    """Use the CPU side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.original_dataloader = dataloader
        self.data = iter(dataloader)

    def next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

    def reset(self):
        self.data = iter(self.original_dataloader)

    def __len__(self) -> int:
        return len(self.original_dataloader)


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.

    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
