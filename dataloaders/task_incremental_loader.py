import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloaders.idataset import DummyArrayDataset
from dataloaders.iq_data_loader import IQDataGenerator
import os


def _normalize_label_array(labels, expected_len, source):
    """Ensure label arrays align with the sample dimension.

    Some of the IQ datasets store labels in a (n_classes, n_samples) layout or
    keep an extra singleton dimension which confuses sklearn utilities during
    validation splits.  This helper reshapes the labels to a 1-D vector of
    length ``expected_len`` and converts one-hot encodings to class indices.
    """

    if labels is None:
        return labels

    arr = np.asarray(labels)
    if arr.ndim == 2 and arr.shape[0] == expected_len and arr.shape[1] == 2:
        return arr.astype(np.int64, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 2 and arr.shape[1] == expected_len:
        return np.ascontiguousarray(arr.T).astype(np.int64, copy=False)
    if arr.size == expected_len:
        arr = arr.reshape(expected_len)
    else:
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.shape and arr.shape[0] != expected_len:
            axis = next((idx for idx, size in enumerate(arr.shape) if size == expected_len), None)
            if axis is None:
                raise ValueError(
                    f"{source} labels have shape {arr.shape}, which is incompatible with "
                    f"{expected_len} samples."
                )
            if axis != 0:
                arr = np.moveaxis(arr, axis, 0)
        if arr.ndim > 1:
            arr = arr.reshape(expected_len, -1)
            if arr.shape[1] == 1:
                arr = arr[:, 0]
            else:
                arr = np.argmax(arr, axis=1)

    return arr.astype(np.int64, copy=False)


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

        self.classes_per_task = []
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

        p_tr, p_te = self.sample_permutations[self._current_task]
        x_train, y_train = self.train_dataset[self._current_task][1][p_tr], self.train_dataset[self._current_task][2][p_tr]
        x_test, y_test = self.test_dataset[self._current_task][1][p_te], self.test_dataset[self._current_task][2][p_te]

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
        def _max_label_value(labels):
            if isinstance(labels, np.ndarray):
                arr = labels
                if arr.ndim == 2 and arr.shape[1] == 2:
                    arr = arr[:, 0]
                arr = arr[arr >= 0]
                return int(arr.max()) if arr.size else -1
            if torch.is_tensor(labels):
                tensor = labels
                if tensor.dim() == 2 and tensor.size(1) == 2:
                    tensor = tensor[:, 0]
                tensor = tensor[tensor >= 0]
                return int(tensor.max().item()) if tensor.numel() > 0 else -1
            return int(np.max(labels))

        if isinstance(self.train_dataset[0][1], np.ndarray):
            n_inputs = self.train_dataset[0][1].shape[1] * (2 if np.iscomplexobj(self.train_dataset[0][1]) else 1)
            n_outputs = 0
            for i in range(len(self.train_dataset)):
                n_outputs = max(n_outputs, _max_label_value(self.train_dataset[i][2]))
                n_outputs = max(n_outputs, _max_label_value(self.test_dataset[i][2]))
            self.n_outputs = n_outputs
            return n_inputs, n_outputs + 1, self.n_tasks
        else:
            n_inputs = self.train_dataset[0][1].size(1)
            n_outputs = 0
            for i in range(len(self.train_dataset)):
                n_outputs = max(n_outputs, _max_label_value(self.train_dataset[i][2]))
                n_outputs = max(n_outputs, _max_label_value(self.test_dataset[i][2]))
            self.n_outputs = n_outputs
            return n_inputs, n_outputs + 1, self.n_tasks

    def get_samples_per_task(self, task_id=None, split="train"):
        if task_id is None:
            task_id = self._current_task
        if self._args.samples_per_task > 0:
            return int(self._args.samples_per_task)
        if split not in ("train", "test"):
            raise ValueError(f"Unknown split '{split}' (expected 'train' or 'test').")
        perms = self.sample_permutations[task_id]
        if isinstance(perms, (list, tuple)):
            idx = 0 if split == "train" else 1
            return int(len(perms[idx]))
        if split == "train":
            return int(len(perms))
        test_data = self.test_dataset[task_id][1]
        if isinstance(test_data, np.ndarray):
            return int(test_data.shape[0])
        return int(test_data.size(0))


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

                x_train = _get(['x_train', 'X_train', 'Xtr', 'xtr', 'Xcv', 'xcv', 'x', 'X'])
                y_train = _get(['y_train', 'Y_train', 'ytr', 'ycv', 'y', 'Y'])
                x_test = _get(['x_test', 'X_test', 'Xte', 'xte'])
                y_test = _get(['y_test', 'Y_test', 'yte'])

                if x_train is not None and y_train is not None:
                    y_train = _normalize_label_array(y_train, x_train.shape[0], f"{fname} train")
                if x_test is not None and y_test is not None:
                    y_test = _normalize_label_array(y_test, x_test.shape[0], f"{fname} test")

                if x_train is None or y_train is None or x_test is None or y_test is None:
                    missing = []
                    training_set = True
                    testing_set = True
                    if x_train is None:
                        missing.append("x_train")
                        training_set = False
                    if y_train is None:
                        missing.append("y_train")
                        training_set = False
                    if training_set:
                        from sklearn.model_selection import train_test_split
                        x_train, x_test, y_train, y_test = train_test_split(
                                                            x_train, y_train, test_size=validation_split, 
                                                            random_state=42, stratify=y_train)
                    if x_test is None:
                        missing.append("x_test")
                        testing_set = False
                    if y_test is None:
                        missing.append("y_test")
                        testing_set = False
                    available = ", ".join(sorted(data.keys()))
                    if not testing_set or not training_set: 
                        raise ValueError(
                            f"Missing dataset entries ({', '.join(missing)}) in {fname}. "
                            f"Available keys: {available}"
                        )
                
                size_tr = x_train.shape[0]
                size_te = min(x_test.shape[0], int(size_tr * validation_split)) if validation_split > 0. else x_test.shape[0]
                x_test = x_test[:size_te]
                y_test = y_test[:size_te]

                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    train_unique = np.unique_counts(y_train[:, 0])
                else:
                    train_unique = np.unique_counts(y_train)
                print(f"Loaded {fname}: Unique train labels: {train_unique}")

                y_train = np.asarray(y_train, dtype=np.int64)
                y_test = np.asarray(y_test, dtype=np.int64)

                # Remap labels to a contiguous global range starting from 0
                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    y_train_cls = y_train[:, 0]
                    y_train_det = y_train[:, 1]
                    y_test_cls = y_test[:, 0]
                    y_test_det = y_test[:, 1]
                    use_detector_arch = bool(getattr(self._args, "use_detector_arch", False))
                    # print(f"Using detector architecture: {use_detector_arch}")
                    has_negatives = (y_train_cls < 0).any() or (y_test_cls < 0).any()

                    unique_labels = np.unique(y_train_cls[y_train_cls >= 0])
                    needs_remap = (
                        unique_labels.size > 0
                        and not np.array_equal(unique_labels, np.arange(unique_labels.size))
                    )
                    y_train_cls_remap = y_train_cls.copy()
                    y_test_cls_remap = y_test_cls.copy()
                    mask_train = y_train_cls >= 0
                    mask_test = y_test_cls >= 0
                    if needs_remap:
                        y_train_cls_remap[mask_train] = (
                            unique_labels.searchsorted(y_train_cls[mask_train]) + labels_offset
                        )
                        y_test_cls_remap[mask_test] = (
                            unique_labels.searchsorted(y_test_cls[mask_test]) + labels_offset
                        )
                    else:
                        y_train_cls_remap[mask_train] = y_train_cls[mask_train] + labels_offset
                        y_test_cls_remap[mask_test] = y_test_cls[mask_test] + labels_offset
                    extra_class = 0
                    if (not use_detector_arch) and has_negatives:
                        extra_class = 1
                        neg_label = labels_offset + unique_labels.size
                        y_train_cls_remap[~mask_train] = neg_label
                        y_test_cls_remap[~mask_test] = neg_label
                    if use_detector_arch:
                        y_train = np.stack([y_train_cls_remap, y_train_det], axis=1)
                        y_test = np.stack([y_test_cls_remap, y_test_det], axis=1)
                    else:
                        y_train = y_train_cls_remap
                        y_test = y_test_cls_remap
                else:
                    use_detector_arch = bool(getattr(self._args, "use_detector_arch", False))
                    has_negatives = (y_train < 0).any() or (y_test < 0).any()
                    unique_labels = np.unique(y_train[y_train >= 0])
                    needs_remap = (
                        unique_labels.size > 0
                        and not np.array_equal(unique_labels, np.arange(unique_labels.size))
                    )
                    y_train_remap = y_train.copy()
                    y_test_remap = y_test.copy()
                    mask_train = y_train >= 0
                    mask_test = y_test >= 0
                    if needs_remap:
                        y_train_remap[mask_train] = (
                            unique_labels.searchsorted(y_train[mask_train]) + labels_offset
                        )
                        y_test_remap[mask_test] = (
                            unique_labels.searchsorted(y_test[mask_test]) + labels_offset
                        )
                    else:
                        y_train_remap[mask_train] = y_train[mask_train] + labels_offset
                        y_test_remap[mask_test] = y_test[mask_test] + labels_offset
                    extra_class = 0
                    if (not use_detector_arch) and has_negatives:
                        extra_class = 1
                        neg_label = labels_offset + unique_labels.size
                        y_train_remap[~mask_train] = neg_label
                        y_test_remap[~mask_test] = neg_label
                    y_train = y_train_remap
                    y_test = y_test_remap
                labels_offset += unique_labels.size + extra_class
                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    remapped = np.unique(y_train[:, 0])
                else:
                    remapped = np.unique(y_train)
                print(f"Loaded {fname}: Remapped labels: {remapped}. Size: {x_train.shape[0]})")

                # 3D array[task, split (xtr/yte/xte/yte), data]
                raw_datasets.append((x_train, y_train, x_test, y_test))
                if y_train.ndim == 2 and y_train.shape[1] == 2:
                    all_labels.append(y_train[:, 0].reshape(-1))
                    all_labels.append(y_test[:, 0].reshape(-1))
                else:
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
                p_tr = np.random.permutation(N)[:n]
                N = self.test_dataset[t][1].shape[0]
                p_te = np.random.permutation(N)[:n]
                self.sample_permutations.append([p_tr, p_te])

            # Track per-task class counts for downstream models.
            def _task_class_count(task):
                labels = task[2]
                if labels.ndim == 2 and labels.shape[1] == 2:
                    labels = labels[:, 0]
                labels = labels[labels >= 0]
                return int(np.unique(labels).size)
            self.classes_per_task = [_task_class_count(task) for task in self.train_dataset]
            print("Built classes_per_task:", self.classes_per_task)
            # Persist on args for convenience.
            self._args.classes_per_task = self.classes_per_task
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
            self.classes_per_task = [
                int(torch.unique(task[2]).numel()) if hasattr(torch, "unique") else len(np.unique(task[2]))
                for task in self.train_dataset
            ]
            self._args.classes_per_task = self.classes_per_task
