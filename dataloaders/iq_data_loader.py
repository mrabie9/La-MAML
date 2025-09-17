import os
from typing import Optional, Tuple

import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def ensure_iq_two_channel(iq_array: np.ndarray) -> np.ndarray:
    """Convert raw IQ samples into a ``(…, 2, L)`` float32 array.

    The datasets we consume may store IQ data in a variety of layouts:

    * complex-valued arrays with shape ``(…, L)``
    * real-valued arrays where I/Q are interleaved along the last axis
      (``[…, 2 * L]``)
    * already separated channel-first or channel-last representations
      (``[…, 2, L]`` or ``[…, L, 2]``)

    This helper normalises the array to the channel-first convention used by
    the models: ``(…, 2, L)`` with ``float32`` dtype.  The function is fully
    vectorised and supports batched inputs directly.
    """

    arr = np.asarray(iq_array)

    if arr.ndim == 0:
        raise ValueError("IQ sample must have at least one dimension")

    if np.iscomplexobj(arr):
        stacked = np.stack((arr.real, arr.imag), axis=-2)
        return np.ascontiguousarray(stacked.astype(np.float32, copy=False))

    arr = arr.astype(np.float32, copy=False)

    # Already channel-first: (..., 2, L)
    if arr.ndim >= 2 and arr.shape[-2] == 2:
        return np.ascontiguousarray(arr)

    # Channel-last: (..., L, 2) -> (..., 2, L)
    if arr.ndim >= 2 and arr.shape[-1] == 2:
        return np.ascontiguousarray(np.swapaxes(arr, -1, -2))

    last_dim = arr.shape[-1]
    if last_dim % 2 != 0:
        raise ValueError(
            "Expected an even number of features to split interleaved IQ data; "
            f"got shape {arr.shape}"
        )

    new_shape = arr.shape[:-1] + (last_dim // 2, 2)
    arr = arr.reshape(new_shape)
    arr = np.swapaxes(arr, -1, -2)
    return np.ascontiguousarray(arr)


class IQDataGenerator(Dataset):
    """Dataset for raw in-phase/quadrature (IQ) samples.

    Parameters
    ----------
    x: np.ndarray
        Complex valued IQ samples. Real and imaginary parts are converted to a
        two channel float representation.
    y: np.ndarray
        Labels for each sample.
    transform: callable, optional
        Optional transform to be applied on a sample.
    convert_to_spectrogram: bool, optional
        If ``True`` the complex sample is converted to a simple frequency
        domain representation using the magnitude of the FFT.  This can be
        useful when models expect spectrogram like inputs.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        transform: Optional[callable] = None,
        convert_to_spectrogram: bool = False,
    ) -> None:
        self.x = x
        self.y = y
        self.transform = transform
        self.convert_to_spectrogram = convert_to_spectrogram

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        iq_sample = self.x[index]
        iq_sample = ensure_iq_two_channel(iq_sample)

        label = self.y[index]

        if self.convert_to_spectrogram:
            iq_sample = self._convert_to_spectrogram(iq_sample)

        if self.transform:
            iq_sample = self.transform(iq_sample)

        return iq_sample, label

    def _convert_to_spectrogram(self, iq_sample: np.ndarray) -> np.ndarray:
        """Convert an IQ sample to a simple spectrogram representation."""
        # Compute magnitude of the FFT for each channel. This is a lightweight
        # approximation of a spectrogram and avoids external dependencies.
        spectrogram = np.abs(np.fft.fft(iq_sample, axis=-1))
        return spectrogram.astype(np.float32)


def load_data_iq(base_path: str, batch_size: int, args=None):
    """Utility to create dataloaders for IQ data.

    The loader expects ``train.npz`` and ``test.npz`` style files inside
    ``base_path``.  Datasets from different sources can be handled by
    inspecting the ``base_path`` name.
    """

    # Load train/test arrays depending on the dataset type
    if "radar" in base_path.lower():
        data = np.load(os.path.join(base_path, "radar_dataset.npz"))
        x_train, y_train = data["xtr"], data["ytr"]
        x_test, y_test = data["xte"], data["yte"]

        scaler_train = preprocessing.StandardScaler().fit(x_train)
        scaler_test = preprocessing.StandardScaler().fit(x_test)
        x_train = scaler_train.transform(x_train)
        x_test = scaler_test.transform(x_test)

    elif "usrp" in base_path.lower():
        train = np.load(os.path.join(base_path, "train.npz"))
        test = np.load(os.path.join(base_path, "test.npz"))
        x_train, y_train = train["X"], train["y"]
        x_test, y_test = test["X"], test["y"]

    elif "rfmls" in base_path.lower():
        train = np.load(os.path.join(base_path, "train.npz"))
        test = np.load(os.path.join(base_path, "test.npz"))
        x_train, y_train = train["X"], train["y"] - 30
        x_test, y_test = test["X"], test["y"] - 30

    else:
        train = np.load(os.path.join(base_path, "train.npz"))
        test = np.load(os.path.join(base_path, "test.npz"))
        x_train, y_train = train["X"], train["y"]
        x_test, y_test = test["X"], test["y"]

    # Split test set into validation and test
    x_test, x_val, y_test, y_val = train_test_split(
        x_test, y_test, test_size=0.7, random_state=42, stratify=y_test
    )

    # Build PyTorch dataloaders
    training_set = IQDataGenerator(x_train, y_train)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0)

    val_set = IQDataGenerator(x_val, y_val)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = IQDataGenerator(x_test, y_test)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    return train_loader, val_loader, test_loader
