"""
Utility functions for image readers
"""

import json
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional, Tuple

import boto3
import matplotlib.pyplot as plt
import numpy as np
import psutil
from czifile.czifile import create_output
from natsort import natsorted
from numpy.typing import ArrayLike

from aind_hcr_data_transformation.models import PathLike


def add_leading_dim(data: ArrayLike) -> ArrayLike:
    """
    Adds a new dimension to existing data.
    Parameters
    ------------------------
    arr: ArrayLike
        Dask/numpy array that contains image data.

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """

    return data[None, ...]


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def extract_data(
    arr: ArrayLike, last_dimensions: Optional[int] = None
) -> ArrayLike:
    """
    Extracts n dimensional data (numpy array or dask array)
    given expanded dimensions.
    e.g., (1, 1, 1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1, 2, 1600, 2000) -> (2, 1600, 2000)

    Parameters
    ------------------------
    arr: ArrayLike
        Numpy or dask array with image data. It is assumed
        that the last dimensions of the array contain
        the information about the image.

    last_dimensions: Optional[int]
        If given, it selects the number of dimensions given
        stating from the end
        of the array
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=3 -> (1, 1600, 2000)
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=1 -> (2000)

    Raises
    ------------------------
    ValueError:
        Whenever the last dimensions value is higher
        than the array dimensions.

    Returns
    ------------------------
    ArrayLike:
        Reshaped array with the selected indices.
    """

    if last_dimensions is not None:
        if last_dimensions > arr.ndim:
            raise ValueError(
                "Last dimensions should be lower than array dimensions"
            )

    else:
        last_dimensions = len(arr.shape) - arr.shape.count(1)

    dynamic_indices = [slice(None)] * arr.ndim

    for idx in range(arr.ndim - last_dimensions):
        dynamic_indices[idx] = 0

    return arr[tuple(dynamic_indices)]


def read_json_as_dict(filepath: PathLike) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def _parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """
    Split an ``s3://bucket/key`` URL into ``(bucket, key)``.

    Parameters
    ----------
    s3_url : str
        S3 URL beginning with ``s3://``.

    Returns
    -------
    tuple of (str, str)
        Bucket name and key. ``key`` may be empty when the URL
        points at a bucket root.
    """
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URL, got: {s3_url!r}")
    path = s3_url[len("s3://"):]
    bucket, _, key = path.partition("/")
    if not bucket:
        raise ValueError(
            f"Could not parse bucket from URL: {s3_url!r}"
        )
    return bucket, key


def sync_dir_to_s3(
    directory_to_upload: PathLike, s3_location: str
) -> None:
    """
    Sync a local directory to an S3 prefix using boto3.

    Walks ``directory_to_upload`` recursively and uploads every
    file to ``<s3_location>/<relative path>``. Objects that already
    exist in S3 with the same size are skipped, mirroring the
    size-based short-circuit of ``aws s3 sync``.

    Parameters
    ----------
    directory_to_upload : PathLike
        Local directory whose contents will be uploaded.
    s3_location : str
        Destination prefix of the form ``s3://bucket/prefix``.

    Returns
    -------
    None
    """
    directory = Path(directory_to_upload)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a directory.")

    bucket, prefix = _parse_s3_url(s3_location)
    prefix = prefix.rstrip("/")

    s3_client = boto3.client("s3")

    # Build map of existing object sizes under the prefix so we can
    # skip files that already match (size-only check).
    existing = {}
    list_prefix = f"{prefix}/" if prefix else ""
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=list_prefix):
        for obj in page.get("Contents") or []:
            existing[obj["Key"]] = obj["Size"]

    files = [p for p in directory.rglob("*") if p.is_file()]
    if not files:
        return

    def _upload(local_path: Path) -> None:
        """Upload one file unless an identically-sized key already exists."""
        rel = local_path.relative_to(directory).as_posix()
        key = f"{prefix}/{rel}" if prefix else rel
        if existing.get(key) == local_path.stat().st_size:
            return
        s3_client.upload_file(str(local_path), bucket, key)

    max_workers = min(8, len(files))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        # Materialize the iterator so exceptions in worker threads
        # propagate to the caller.
        for _ in pool.map(_upload, files):
            pass


def copy_file_to_s3(
    file_to_upload: PathLike, s3_location: str
) -> None:
    """
    Upload a single local file to S3 using boto3.

    When ``s3_location`` ends with ``/`` (or has an empty key) the
    file is uploaded under that prefix using its basename;
    otherwise it is uploaded to the exact key given. This mirrors
    ``aws s3 cp`` semantics.

    Parameters
    ----------
    file_to_upload : PathLike
        Local file path.
    s3_location : str
        Destination URL of the form ``s3://bucket/key`` or
        ``s3://bucket/prefix/``.

    Returns
    -------
    None
    """
    file_path = Path(file_to_upload)
    if not file_path.is_file():
        raise FileNotFoundError(f"{file_path} is not a file.")

    bucket, key = _parse_s3_url(s3_location)
    if not key or s3_location.endswith("/"):
        base = key.rstrip("/")
        key = f"{base}/{file_path.name}" if base else file_path.name

    s3_client = boto3.client("s3")
    s3_client.upload_file(str(file_path), bucket, key)


def get_available_cpu_count() -> int:
    """
    Return the number of CPUs actually available to this process.

    Resolution order:

    1. ``SLURM_CPUS_PER_TASK`` (set by SLURM when ``--cpus-per-task``
       is requested).
    2. ``os.sched_getaffinity(0)`` (Linux; honors cgroup/taskset/SLURM
       affinity masks).
    3. ``multiprocessing.cpu_count()`` (host total; only used as a last
       resort because it ignores SLURM allocation and reports the full
       node).

    Returns
    -------
    int
        Number of CPUs the process should plan to use. Always >= 1.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, multiprocessing.cpu_count())


def validate_slices(start_slice: int, end_slice: int, len_dir: int):
    """
    Validates that the slice indices are within bounds

    Parameters
    ----------
    start_slice: int
        Start slice integer

    end_slice: int
        End slice integer

    len_dir: int
        Len of czi directory
    """
    if not (0 <= start_slice < end_slice <= len_dir):
        msg = (
            f"Slices out of bounds. Total: {len_dir}"
            f"Start: {start_slice}, End: {end_slice}"
        )
        raise ValueError(msg)


def parallel_reader(
    args: tuple,
    out: np.ndarray,
    nominal_start: np.ndarray,
    start_slice: int,
    ax_index: int,
    resize: bool,
    order: int,
):
    """
    Reads a single subblock and places it in the output array.

    Parameters
    ----------
    args: tuple
        Index and directory entry of the czi file.

    out: np.ndarray
        Placeholder array for the data

    nominal_start: np.ndarray
        Nominal start of the dataset when it was acquired.

    start_slice: int
        Start slice.

    ax_index: int
        Axis index.

    resize: bool
        True if resizing is needed when reading CZI data.

    order: int
        Interpolation in resizing.
    """
    idx, directory_entry = args
    subblock = directory_entry.data_segment()
    tile = subblock.data(resize=resize, order=order)
    dir_start = np.array(directory_entry.start) - nominal_start

    # Calculate index placement
    index = tuple(slice(i, i + k) for i, k in zip(dir_start, tile.shape))
    index = list(index)
    index[ax_index] = slice(
        index[ax_index].start - start_slice, index[ax_index].stop - start_slice
    )

    try:
        out[tuple(index)] = tile
    except ValueError as e:
        raise ValueError(f"Error writing subblock {idx + start_slice}: {e}")


def read_slices_czi(
    czi_stream,
    subblock_directory: List,
    start_slice: int,
    end_slice: int,
    slice_axis: Optional[str] = "z",
    resize: Optional[bool] = True,
    order: Optional[int] = 0,
    out: Optional[List[int]] = None,
    max_workers: Optional[int] = None,
):
    """
    Reads chunked data from CZI files. From AIND-Zeiss
    the data is being chunked in a slice basis. Therefore,
    we assume the slice axis to be 'z'.

    Parameters
    ----------
    czi_stream
        Opened CZI file decriptor.

    subblock_directory: List
        List of subblock directories. These must be ordered.

    start_slice: int
        Start slice from where the data will be pulled.

    end_slice: int
        End slice from where the data will be pulled.

    slice_axis: Optional[str] = 'z'
        Axis in which start and end slice parameters will
        be applied.
        Default: 'z'

    resize: Optional[bool] = True
        If we want to resize the tile from the CZI file.
        Default: True

    order: Optional[int] = 0
        Interpolation order
        Default: 0

    out: Optional[List[int]] = None
        Out shape of the final array
        Default: None

    max_workers: Optional[int] = None
        Number of workers that will be pulling data. When ``None``
        (default), the pool is auto-sized from the process's actual CPU
        allocation via :func:`get_available_cpu_count` (which prefers
        ``SLURM_CPUS_PER_TASK`` and ``os.sched_getaffinity`` over the
        host-wide ``multiprocessing.cpu_count``). Set to ``1`` to force
        a fully serial reader, which sidesteps a known thread-safety
        bug in ``czifile`` that can segfault under concurrent subblock
        reads.
        Default: None

    Returns
    -------
    np.ndarray
        Numpy array with the pulled data
    """

    shape, dtype, axes = (
        czi_stream.shape,
        czi_stream.dtype,
        list(czi_stream.axes.lower()),
    )
    nominal_start = np.array(czi_stream.start)

    len_dir = len(subblock_directory)

    validate_slices(start_slice, end_slice, len_dir)

    ax_index = axes.index(slice_axis.lower())
    new_shape = list(shape)
    new_shape[ax_index] = end_slice - start_slice
    new_shape[axes.index("c")] = 1  # Assume 1 channel per CZI

    out = create_output(out, new_shape, dtype)
    if max_workers is None:
        # Use the SLURM allocation (or cgroup affinity) rather than the
        # host's total CPU count, which on shared nodes can be dozens of
        # cores. Over-sizing the pool both ignores the resource manager's
        # allocation and amplifies the known ``czifile`` thread-safety
        # bug under concurrent subblock reads.
        max_workers = min(
            get_available_cpu_count(), end_slice - start_slice
        )
    max_workers = max(1, max_workers)

    selected_entries = subblock_directory[start_slice:end_slice]

    if max_workers > 1 and end_slice - start_slice > 1:
        czi_stream._fh.lock = True
        with ThreadPoolExecutor(max_workers) as executor:
            executor.map(
                lambda args: parallel_reader(
                    args,
                    out,
                    nominal_start,
                    start_slice,
                    ax_index,
                    resize,
                    order,
                ),
                enumerate(selected_entries),
            )
        czi_stream._fh.lock = None
    else:
        for idx, entry in enumerate(selected_entries):
            parallel_reader(
                (idx, entry),
                out,
                nominal_start,
                start_slice,
                ax_index,
                resize,
                order,
            )

    if hasattr(out, "flush"):
        out.flush()

    return np.squeeze(out)


def generate_jumps(n: int, jump_size: Optional[int] = 128):
    """
    Generates jumps for indexing.

    Parameters
    ----------
    n: int
        Final number for indexing.
        It is exclusive in the final number.

    jump_size: Optional[int] = 128
        Jump size.
    """
    jumps = list(range(0, n, jump_size))
    # if jumps[-1] + jump_size >= n:
    #     jumps.append(n)

    return jumps


def get_axis_index(czi_shape: List[int], czi_axis: int, axis_name: str):
    """
    Gets the axis index from the CZI natural shape.

    Parameters
    ----------
    czi_shape: List[int]
        List of ints of the CZI shape. CZI files come
        with many more axis than traditional file formats.
        Please, check its documentation.

    czi_axis: int
        Axis from which we will pull the index.

    axis_name: str
        Axis name. Allowed axis names are:
        ['b', 'v', 'i', 'h', 'r', 's', 'c', 't', 'z', 'y', 'x', '0']
    """
    czi_axis = list(str(czi_axis).lower())
    axis_name = axis_name.lower()
    ALLOWED_AXIS_NAMES = [
        "b",
        "v",
        "i",
        "h",
        "r",
        "s",
        "c",
        "t",
        "z",
        "y",
        "x",
        "0",
    ]

    if axis_name not in ALLOWED_AXIS_NAMES:
        raise ValueError(f"Axis {axis_name} not valid!")

    czi_shape = list(czi_shape)
    ax_index = czi_axis.index(axis_name)

    return ax_index, czi_shape[ax_index]


def czi_block_generator(
    czi_decriptor,
    axis_jumps: Optional[int] = 128,
    slice_axis: Optional[str] = "z",
    max_workers: Optional[int] = None,
):
    """
    CZI data block generator.

    Parameters
    ----------
    czi_decriptor
        Opened CZI file.

    axis_jumps: int
        Number of jumps in a given axis.
        Default: 128

    slice_axis: str
        Axis in which the jumps will be
        generated.
        Default: 'z'

    max_workers: Optional[int] = None
        Maximum number of threads used by the underlying
        ``read_slices_czi`` call. When ``None`` (default), the reader
        auto-sizes the thread pool from the process's CPU allocation
        (``SLURM_CPUS_PER_TASK`` / cgroup affinity / host CPU count, in
        that order) rather than the host's full CPU count. Set to ``1``
        to force a fully serial reader, which sidesteps a known
        thread-safety bug in ``czifile`` that can segfault under
        concurrent subblock reads.

    Yields
    ------
    np.ndarray
        Numpy array with the data
        of the picked block.

    slice
        Slice of start and end positions
        in a given axis.
    """

    axis_index, axis_shape = get_axis_index(
        czi_decriptor.shape, czi_decriptor.axes, slice_axis
    )

    subblock_directory = czi_decriptor.filtered_subblock_directory

    # Sorting indices so planes are ordered
    ordered_subblock_directory = natsorted(
        subblock_directory, key=lambda sb: sb.start[axis_index]
    )

    jumps = generate_jumps(axis_shape, axis_jumps)
    n_jumps = len(jumps)
    for i, start_slice in enumerate(jumps):
        if i + 1 < n_jumps:
            end_slice = jumps[i + 1]

        else:
            end_slice = axis_shape

        block = read_slices_czi(
            czi_decriptor,
            subblock_directory=ordered_subblock_directory,
            start_slice=start_slice,
            end_slice=end_slice,
            slice_axis=slice_axis,
            resize=True,
            order=0,
            out=None,
            max_workers=max_workers,
        )
        yield block, slice(start_slice, end_slice)


def write_json(
    output_path: str,
    json_data: dict,
    bucket_name: Optional[str] = None,
):
    """
    Writes the multiscale json in the top
    level directory of the zarr.

    Parameters
    ----------
    output_path: str
        Output path where we want the json

    json_data: dict
        Dictionary with the zarr.json metadata.

    bucket_name: Optional[str]
        Path where we want to store the json in s3.
        If default is None, the file will be saved
        locally. Default: None

    """
    json_key = f"{output_path}/zarr.json"
    if bucket_name:
        s3 = boto3.client("s3")

        # Upload the JSON string as a file to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=json_key,
            Body=json.dumps(json_data, indent=2),
            ContentType="application/json",
        )

    else:
        with open(json_key, "w") as fp:
            json.dump(json_data, fp, indent=2)


class MemoryLogger:
    """
    Logs memory and CPU usage of the current process.
    Can be used as a context manager or via static methods.
    """

    def __init__(
        self,
        label="MemoryLogger",
        interval: Optional[float] = None,
        logger=None,
    ):
        """
        Parameters
        ----------:
            label (str): Label to include in log messages.
            interval (float): If set, logs usage every `interval`
             seconds in a background thread.
            logger (logging.Logger): Optional custom logger.
        """
        self.label = label
        self.interval = interval
        self.logger = logger or logging.getLogger(label)
        self._stop_event = threading.Event()
        self._thread = None
        self.timestamps: List[float] = []
        self.memory_mb: List[float] = []
        self.cpu_percent: List[float] = []

    @staticmethod
    def log_memory_cpu(label="MemoryLogger", logger=None):
        """Log current process memory and CPU usage once.
        Parameters:
        ----------
            label (str): Label to include in log messages.

        logger (logging.Logger): Optional custom logger.
        Returns:
        -------
            tuple: (timestamp, memory in MB, CPU percent)

        """
        logger = logger or logging.getLogger(label)
        process = psutil.Process(os.getpid())
        mem = process.memory_full_info()
        cpu = process.cpu_percent(interval=0.1)
        logger.info(
            f"[{label}] RSS={mem.rss/1e6:.2f}MB, "
            f"USS={mem.uss/1e6:.2f}MB, "
            f"VMS={mem.vms/1e6:.2f}MB, CPU={cpu:.1f}%"
        )
        return time.time(), mem.rss / 1e6, cpu

    def _background_log(self):
        """Background thread to log memory and CPU usage at regular intervals.
        Parameters:
        ----------
            None
        """

        while not self._stop_event.is_set():
            t, mem, cpu = self.log_memory_cpu(self.label, self.logger)
            self.timestamps.append(t)
            self.memory_mb.append(mem)
            self.cpu_percent.append(cpu)
            time.sleep(self.interval)

    def __enter__(self):
        """Context manager entry point.
        Parameters:
        ----------
            None

        Returns:
        -------
            self: MemoryLogger instance
        """
        t, mem, cpu = self.log_memory_cpu(self.label, self.logger)
        self.timestamps.append(t)
        self.memory_mb.append(mem)
        self.cpu_percent.append(cpu)
        if self.interval:
            self._thread = threading.Thread(
                target=self._background_log, daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point.
        Parameters:
        ----------
            exc_type: Exception type
            exc_val: Exception value
            exc_tb: Exception traceback

        Returns:
        -------
            None
        """

        if self.interval:
            self._stop_event.set()
            self._thread.join()
        t, mem, cpu = self.log_memory_cpu(self.label, self.logger)
        self.timestamps.append(t)
        self.memory_mb.append(mem)
        self.cpu_percent.append(cpu)

    def plot(self, save_path: str = "memory_cpu_profile.png"):
        """
        Plot memory and CPU usage over time and save to file.

        Parameters
        ----------
            save_path (str): Path to save the plot image.

        """
        if not self.timestamps:
            raise RuntimeError(
                "No data to plot. Use as context manager with interval."
            )
        rel_time = [t - self.timestamps[0] for t in self.timestamps]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.plot(rel_time, self.memory_mb, "b-", label="Memory (MB)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Memory (MB)", color="b")
        ax2 = ax1.twinx()
        ax2.plot(rel_time, self.cpu_percent, "r-", label="CPU (%)")
        ax2.set_ylabel("CPU (%)", color="r")
        plt.title(self.label)
        fig.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        self.logger.info(f"Memory/CPU profile plot saved to {save_path}")
