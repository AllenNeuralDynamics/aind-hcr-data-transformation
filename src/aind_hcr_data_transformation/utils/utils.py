"""
Utility functions for image readers
"""

import json
import multiprocessing
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
from czifile.czifile import create_output

from aind_hcr_data_transformation.models import ArrayLike, PathLike


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


def sync_dir_to_s3(directory_to_upload: PathLike, s3_location: str) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    directory_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "sync",
        str(directory_to_upload),
        s3_location,
        "--only-show-errors",
    ]

    subprocess.run(base_command, shell=shell, check=True)


def copy_file_to_s3(file_to_upload: PathLike, s3_location: str) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    file_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "cp",
        str(file_to_upload),
        s3_location,
        "--only-show-errors",
    ]

    subprocess.run(base_command, shell=shell, check=True)


def read_slices_czi(
    czi_stream,
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
        Number of workers that will be pulling data.
        Default: None

    Returns
    -------
    np.ndarray
        Numpy array with the pulled data
    """
    shape = czi_stream.shape
    dtype = czi_stream.dtype
    axes = list(czi_stream.axes.lower())

    slice_axis = str(slice_axis).lower()

    # print(axes, len(axes), len(shape), shape, axes.index(slice_axis))
    nominal_start = np.array(czi_stream.start)

    subblock_directory = czi_stream.filtered_subblock_directory
    len_dir = len(subblock_directory)

    # Validate slice parameters
    if start_slice > len_dir or end_slice > len_dir:
        raise ValueError(
            f"Slices out of bounds. Total: {len_dir} - Start: {start_slice} end {end_slice}"
        )

    if start_slice < 0 or end_slice < 0:
        raise ValueError(
            f"Slices out of bounds. Total: {len_dir} - Start: {start_slice} end {end_slice}"
        )

    if start_slice >= end_slice:
        raise ValueError("Start and end slices can't be the same range!")

    # Calculate the shape of the output array
    new_shape = list(shape)
    ax_index = axes.index(slice_axis)
    new_shape[ax_index] = end_slice - start_slice

    # Setting channel axis to 1 by default - Zeiss assuming 1 channel per czi
    new_shape[axes.index("c")] = 1

    # Create output array if not provided
    out = create_output(out, new_shape, dtype)

    # Set maximum number of worker threads
    if max_workers is None:
        max_workers = min(
            multiprocessing.cpu_count() // 2, end_slice - start_slice
        )

    # Filter directory entries to only process the requested range
    selected_entries = subblock_directory[start_slice:end_slice]

    def parallel_reader(args):
        """
        Parallel CZI internal reader
        
        Parameters
        ----------
        args: tuple
            Index and directory entry of the CZI file.
        """
        idx, directory_entry = args

        subblock = directory_entry.data_segment()
        tile = subblock.data(resize=resize, order=order)
        dir_start = tuple(np.array(directory_entry.start) - nominal_start)

        # Calculate the index for placing this tile in the output array
        index = list(slice(i, i + k) for i, k in zip(dir_start, tile.shape))
        index[ax_index] = slice(
            index[ax_index].start - start_slice,
            index[ax_index].stop - start_slice,
        )

        index = tuple(index)

        # Print the index information
        # print(f"Subblock {idx + start_slice}")
        # print(f"  Directory entry start: {directory_entry.start}")
        # print(f"  Tile shape: {tile.shape}")
        # print(f"  Index used: {index}")

        try:
            out[index] = tile
        except ValueError as e:
            raise ValueError(
                f"Error writing subblock {idx + start_slice}: {e}"
            )

        return idx

    # Process the subblocks
    if max_workers > 1 and end_slice - start_slice > 1:
        czi_stream._fh.lock = True
        with ThreadPoolExecutor(max_workers) as executor:
            results = list(executor.map(parallel_reader, enumerate(selected_entries)))
        czi_stream._fh.lock = None
        # print(f"Processed {len(results)} subblocks using {max_workers} workers")
    else:
        for idx, directory_entry in enumerate(selected_entries):
            parallel_reader((idx, directory_entry))
        # print(f"Processed {len(selected_entries)} subblocks sequentially")

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

    jumps = generate_jumps(axis_shape, axis_jumps)
    n_jumps = len(jumps)
    for i, start_slice in enumerate(jumps):
        if i + 1 < n_jumps:
            end_slice = jumps[i + 1] - 1

        else:
            end_slice = axis_shape

        block = read_slices_czi(
            czi_decriptor,
            start_slice=start_slice,
            end_slice=end_slice,
            slice_axis=slice_axis,
            resize=True,
            order=0,
            out=None,
            max_workers=None,
        )
        yield block, slice(start_slice, end_slice)
