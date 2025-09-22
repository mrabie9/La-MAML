import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets

from dataloaders.idataset import DummyArrayDataset
from dataloaders.iq_data_loader import IQDataGenerator
import os


class IncrementalLoader:

    def __init__(
        self,
        args,
        shuffle=True,
        seed=1,
    ):
        self._args = args
        validation_split=args.validation
        increment=args.increment

        self._setup_data(
            class_order_type=args.class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )

        self._current_task = 0

        self._batch_size = args.batch_size
        self._test_batch_size = args.test_batch_size        
        self._workers = args.workers
        self._shuffle = shuffle

        self._setup_test_tasks()

    @property
    def n_tasks(self):
        return len(self.test_dataset)
    
    def new_task(self):
        if self._current_task >= len(self.test_dataset):
            raise Exception("No more tasks.")

        p = self.sample_permutations[self._current_task]
        x_train, y_train = self.train_dataset[self._current_task][1][p], self.train_dataset[self._current_task][2][p]
        x_test, y_test = self.test_dataset[self._current_task][1][p], self.test_dataset[self._current_task][2][p]

        train_loader = self._get_loader(x_train, y_train, mode="train")
        test_loader = self._get_loader(x_test, y_test, mode="test")

        task_info = {
            "min_class": 0,
            "max_class": self.n_outputs,
            "increment": -1,
            "task": self._current_task,
            "max_task": len(self.test_dataset),
            "n_train_data": len(x_train),
            "n_test_data": len(x_test)
        }

        self._current_task += 1

        return task_info, train_loader, None, test_loader

    def _setup_test_tasks(self):
        self.test_tasks = []
        for i in range(len(self.test_dataset)):
            # .append(x, y, mode="test")
            self.test_tasks.append(self._get_loader(self.test_dataset[i][1], self.test_dataset[i][2], mode="test"))

    def get_tasks(self, dataset_type='test'):
        if dataset_type == 'test':
            return self.test_dataset
        elif dataset_type == 'val':
            return self.test_dataset
        else:
            raise NotImplementedError("Unknown mode {}.".format(dataset_type))

    def get_dataset_info(self):
        if isinstance(self.train_dataset[0][1], np.ndarray):
            n_inputs = self.train_dataset[0][1].shape[1] * (2 if np.iscomplexobj(self.train_dataset[0][1]) else 1)
            n_outputs = 0
            for i in range(len(self.train_dataset)):
                n_outputs = max(n_outputs, int(self.train_dataset[i][2].max()))
                n_outputs = max(n_outputs, int(self.test_dataset[i][2].max()))
            self.n_outputs = n_outputs
            return n_inputs, n_outputs + 1, self.n_tasks
        else:
            n_inputs = self.train_dataset[0][1].size(1)
            n_outputs = 0
            for i in range(len(self.train_dataset)):
                n_outputs = max(n_outputs, self.train_dataset[i][2].max())
                n_outputs = max(n_outputs, self.test_dataset[i][2].max())
            self.n_outputs = n_outputs
            return n_inputs, n_outputs.item()+1, self.n_tasks


    def _get_loader(self, x, y, shuffle=True, mode="train"):
        if mode == "train":
            batch_size = self._batch_size
        elif mode == "test":
            batch_size = self._test_batch_size
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        if isinstance(x, np.ndarray):
            dataset = IQDataGenerator(x, y)
        else:
            dataset = DummyArrayDataset(x, y)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0#self._workers
        )


    def _setup_data(self, class_order_type=False, seed=1, increment=10, validation_split=0.):
        # FIXME: handles online loading of images
        torch.manual_seed(seed)

        data_files = [f for f in os.listdir(self._args.data_path) if f.endswith('.npz')]
        if data_files and self._args.dataset.lower() == 'iq':
            data_files.sort()
            raw_datasets = []
            all_labels = []
            labels_offset = 0

            # Load npz files and concatenate datasets
            for fname in data_files:
                data = np.load(os.path.join(self._args.data_path, fname))

                def _get(keys):
                    for k in keys:
                        if k in data:
                            return data[k]
                    return None

                x_train = _get(['x_train', 'X_train', 'Xtr', 'X'])
                y_train = _get(['y_train', 'Y_train', 'ytr', 'y'])
                x_test = _get(['x_test', 'X_test', 'Xte', 'X'])
                y_test = _get(['y_test', 'Y_test', 'yte', 'y'])

                if y_train is None or y_test is None:
                    raise ValueError(f"Labels not found in {fname}")

                y_train = np.asarray(y_train, dtype=np.int64)
                y_test = np.asarray(y_test, dtype=np.int64)

                # Remap labels to a contiguous range starting from 0
                unique_labels = np.unique(y_train)
                needs_remap = not np.array_equal(unique_labels, np.arange(unique_labels.size))
                if needs_remap:
                    y_train = unique_labels.searchsorted(y_train) + labels_offset
                    y_test = unique_labels.searchsorted(y_test) + labels_offset
                    labels_offset += unique_labels.size

                # 3D array[task, split (xtr/yte/xte/yte), data]
                raw_datasets.append((x_train, y_train, x_test, y_test))
                all_labels.append(y_train.reshape(-1))
                all_labels.append(y_test.reshape(-1))

            if not raw_datasets:
                raise ValueError("No IQ datasets were loaded. Please check the data path.")

            self.train_dataset, self.test_dataset = [], []
            for x_train, y_train, x_test, y_test in raw_datasets:
                self.train_dataset.append((None, x_train, y_train.astype(np.int64)))
                self.test_dataset.append((None, x_test, y_test.astype(np.int64)))

            self.sample_permutations = []
            for t in range(len(self.train_dataset)):
                N = self.train_dataset[t][1].shape[0] # number of samples in task t
                if self._args.samples_per_task <= 0:
                    n = N
                else:
                    n = min(self._args.samples_per_task, N)
                # randomly shuffle data
                p = np.random.permutation(N)[:n]
                self.sample_permutations.append(p)
        else:
            self.train_dataset, self.test_dataset = torch.load(os.path.join(self._args.data_path, self._args.dataset + ".pt"))

            self.sample_permutations = []

            # for every task, accumulate a shuffled set of samples_per_task
            for t in range(len(self.train_dataset)):
                N = self.train_dataset[t][1].size(0)
                if self._args.samples_per_task <= 0:
                    n = N
                else:
                    n = min(self._args.samples_per_task, N)

                p = torch.randperm(N)[0:n]
                self.sample_permutations.append(p)
