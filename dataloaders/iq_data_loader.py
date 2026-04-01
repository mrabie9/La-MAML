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
    if arr.ndim > 2 and arr.shape[-2] == 2:
        return np.ascontiguousarray(arr)

    # Channel-last: (..., L, 2) -> (..., 2, L)
    if arr.ndim > 2 and arr.shape[-1] == 2:
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


def iq_numpy_batch_to_three_adc_channel_first(x: np.ndarray) -> np.ndarray:
    """Reshape batched IQ features to ``(N, 3, L)`` float32 (ADC0 = I/Q interleaved).

    This matches the tensor layout produced by :class:`IQDataGenerator` when
    ``target_adc_channels=3``: row 0 carries interleaved I and Q along length ``L``;
    rows 1 and 2 are zeros. The ResNet1D 3-ADC path then views each row as
    interleaved pairs: ``(N, 3, L)`` -> ``(N, 3, 2, L//2)``.

    Args:
        x: Batch with one of the following shapes:
            ``(N, F)`` — flat interleaved I/Q (``F`` even);
            ``(N, 2, L)`` — channel-first I and Q;
            ``(N, L, 2)`` — channel-last I/Q;
            ``(N, 3, L)`` — already 3 ADC rows (returned as contiguous float32).

    Returns:
        Array of shape ``(N, 3, L)`` with ``L`` even when starting from flat or
        2-channel inputs.

    Raises:
        ValueError: If ``x`` has an unsupported rank or an odd flat length.

    Usage:
        >>> x_flat = np.zeros((4, 1024), dtype=np.float32)
        >>> y = iq_numpy_batch_to_three_adc_channel_first(x_flat)
        >>> y.shape
        (4, 3, 1024)
    """
    arr = np.asarray(x)
    if arr.ndim == 2:
        batch, features = arr.shape
        if batch == 0:
            return np.zeros((0, 3, 0), dtype=np.float32)
        if features % 2 != 0:
            raise ValueError(
                "Flat IQ requires an even feature count for I/Q pairs; "
                f"got shape {tuple(arr.shape)}"
            )
        arr = arr.astype(np.float32, copy=False)
        i_ch = arr[:, 0::2]
        q_ch = arr[:, 1::2]
        interleaved = np.empty((batch, features), dtype=np.float32)
        interleaved[:, 0::2] = i_ch
        interleaved[:, 1::2] = q_ch
        out = np.zeros((batch, 3, features), dtype=np.float32)
        out[:, 0, :] = interleaved
        return np.ascontiguousarray(out)

    if arr.ndim == 3:
        if arr.shape[0] == 0:
            return np.zeros((0, 3, 0), dtype=np.float32)
        # Channel-last (N, L, 2) or (N, L, 3)
        if arr.shape[-1] == 2 and arr.shape[-2] != 2:
            arr = np.ascontiguousarray(
                np.swapaxes(arr.astype(np.float32, copy=False), 1, 2)
            )
            return iq_numpy_batch_to_three_adc_channel_first(arr)

        if arr.shape[1] == 3:
            return np.ascontiguousarray(arr.astype(np.float32, copy=False))

        if arr.shape[1] == 2:
            arr = arr.astype(np.float32, copy=False)
            n_batch, _two, length = arr.shape
            interleaved = np.empty((n_batch, 2 * length), dtype=np.float32)
            interleaved[:, 0::2] = arr[:, 0, :]
            interleaved[:, 1::2] = arr[:, 1, :]
            out = np.zeros((n_batch, 3, 2 * length), dtype=np.float32)
            out[:, 0, :] = interleaved
            return np.ascontiguousarray(out)

    raise ValueError(
        "iq_numpy_batch_to_three_adc_channel_first expects x of shape "
        f"(N, F), (N, 2, L), (N, L, 2), or (N, 3, L); got {tuple(arr.shape)}"
    )


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
        target_adc_channels: int | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.transform = transform
        self.convert_to_spectrogram = convert_to_spectrogram
        self.target_adc_channels = target_adc_channels

    def __len__(self) -> int:  # pragma: no cover - simple wrapper
        return self.x.shape[0]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        iq_sample = self.x[index, :]

        if iq_sample.ndim == 2:
            if iq_sample.shape[0] in (2, 3):
                iq_sample = iq_sample.astype(np.float32, copy=False)
            elif iq_sample.shape[1] in (2, 3):
                iq_sample = np.swapaxes(iq_sample, 0, 1).astype(np.float32, copy=False)
            else:
                flat = iq_sample.reshape(-1)
                i = flat[0::2]
                q = flat[1::2]
                iq_sample = np.stack([i, q], axis=0).astype(np.float32)
        else:
            # Represent complex input as two channels: I and Q
            if np.iscomplexobj(iq_sample):
                i = iq_sample.real
                q = iq_sample.imag
            else:
                i = iq_sample[0::2]
                q = iq_sample[1::2]

            iq_sample = np.stack([i, q], axis=0).astype(np.float32)

        if self.target_adc_channels == 3:
            # Standardize to a 3-ADC interleaved representation expected by the
            # 3-channel input path in `ResNet1D._prepare_input`, i.e. `(3, 2*L)`
            # with ADC0 holding the original I/Q and ADC1/ADC2 set to zeros.
            if iq_sample.shape[0] == 2:
                l_iq = iq_sample.shape[1]
                interleaved = np.empty((2 * l_iq,), dtype=np.float32)
                interleaved[0::2] = iq_sample[0]
                interleaved[1::2] = iq_sample[1]
                padded = np.zeros((3, interleaved.shape[0]), dtype=np.float32)
                padded[0, :] = interleaved
                iq_sample = padded
            elif iq_sample.shape[0] == 3:
                iq_sample = iq_sample.astype(np.float32, copy=False)
            else:
                raise ValueError(
                    f"Expected I/Q channels (2) or ADC channels (3) for target_adc_channels=3; "
                    f"got shape {tuple(iq_sample.shape)}."
                )

        label = self.y[index]

        if self.convert_to_spectrogram:
            iq_sample = self._convert_to_spectrogram(iq_sample)

        if self.transform:
            iq_sample = self.transform(iq_sample)

        if isinstance(label, (list, tuple)) and len(label) == 2:
            return iq_sample, (label[0], label[1])
        if isinstance(label, np.ndarray) and label.ndim == 1 and label.shape[0] == 2:
            return iq_sample, (label[0], label[1])
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
        x_train, y_train = train["X"], train["y"]
        x_test, y_test = test["X"], test["y"]

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
    train_loader = DataLoader(
        training_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    val_set = IQDataGenerator(x_val, y_val)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)

    test_set = IQDataGenerator(x_test, y_test)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=0
    )

    return train_loader, val_loader, test_loader
