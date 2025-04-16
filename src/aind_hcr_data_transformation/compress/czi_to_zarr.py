"""
CZI to Zarr writer. It takes an input path
where 3D stacks are located, then these
stacks are loaded into memory and written
to zarr.
"""

import asyncio
import logging
import os
import time
from typing import (
    Any,
    Dict,
    Hashable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
    Union,
    cast,
)

import czifile
import dask
import dask.array as da
import numpy as np
import tensorstore as ts
import xarray_multiscale
import zarr
from ome_zarr.format import CurrentFormat
from ome_zarr.writer import (
    AxesType,
    Format,
    JSONDict,
    _get_valid_axes,
    _validate_datasets,
    write_multiscales_metadata,
)

from aind_hcr_data_transformation.utils.utils import (
    czi_block_generator,
    pad_array_n_d,
)


def create_spec(
    output_path: str,
    data_shape: list,
    data_dtype: str,
    shard_shape: list,
    chunk_shape: list,
    zyx_resolution: list,
    compressor_kwargs: dict,
    scale: str = "0",
    cpu_cnt: int = None,
    aws_region: str = "us-west-2",
    bucket_name: str = None,
    aws_profile: str = "default",
    credentials_file: str = "~/.aws/credentials",
    read_cache_bytes: int = 1 << 30,
) -> dict:
    """
    Create a TensorStore Zarr v3 specification for writing to an S3-backed dataset.

    Parameters
    ----------
    output_path : str
        Path inside the S3 bucket where the dataset will be stored.
    data_shape : list of int
        Shape of the full dataset in [t, c, z, y, x] order.
    data_dtype : str
        Data type of the dataset (e.g., 'uint16', 'float32').
    shard_shape : list of int
        Shape of the sharded outer chunks.
    chunk_shape : list of int
        Shape of the internal compressed chunks inside each shard.
    zyx_resolution : list
        Spatial resolution in microns for the z, y, x axes.
    compressor_kwargs: dict
        Compressor parameters for tensorstore
    scale : str, optional
        Scale level identifier (e.g., "0", "1", etc.). Default is "0".
    cpu_cnt : int, optional
        Number of CPU threads to use. Defaults to the number of system CPUs.
    aws_region : str, optional
        AWS region where the S3 bucket resides. Default is "us-west-2".
    bucket_name : str, optional
        Name of the S3 bucket. Default is "aind-msma-morphology-data".
    aws_profile : str, optional
        AWS CLI profile to use for authentication. Default is "default".
    credentials_file : str, optional
        Path to AWS credentials file. Default is "~/.aws/credentials".
    read_cache_bytes : int, optional
        Size of the read cache pool in bytes. Default is 1GB.

    Returns
    -------
    spec : dict
        TensorStore specification dictionary to create a new Zarr v3 dataset.
    """
    if cpu_cnt is None:
        cpu_cnt = multiprocessing.cpu_count()

    zyx_resolution = [
        f"{r}um" if r is not None else None for r in zyx_resolution
    ]
    zyx_resolution = [None] * (5 - len(zyx_resolution)) + zyx_resolution

    kvstore_dict = {
        "driver": "file",
    }

    if bucket_name is not None:
        kvstore_dict = {
            "driver": "s3",
            "bucket": bucket_name,
            "aws_region": aws_region,
            "aws_credentials": {
                "type": "profile",
                "profile": aws_profile,
                "credentials_file": os.path.expanduser(credentials_file),
            },
            "context": {
                "cache_pool": {"total_bytes_limit": read_cache_bytes},
                "data_copy_concurrency": {"limit": cpu_cnt},
                "s3_request_concurrency": {"limit": cpu_cnt},
                "experimental_s3_rate_limiter": {
                    "read_rate": cpu_cnt,
                    "write_rate": cpu_cnt,
                },
            },
        }

    return {
        "driver": "zarr3",
        "kvstore": {
            **kvstore_dict,
            "path": output_path,
        },
        "path": str(scale),
        "recheck_cached_metadata": False,
        "recheck_cached_data": False,
        "metadata": {
            "shape": data_shape,
            "zarr_format": 3,
            "node_type": "array",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": shard_shape},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "attributes": {
                "dimension_units": zyx_resolution,
            },
            "dimension_names": ["t", "c", "z", "y", "x"],
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {
                                "name": "blosc",
                                "configuration": compressor_kwargs,
                            },
                        ],
                        "index_codecs": [
                            {
                                "name": "bytes",
                                "configuration": {"endian": "little"},
                            },
                            {"name": "crc32c"},
                        ],
                        "index_location": "end",
                    },
                }
            ],
            "data_type": data_dtype,
        },
        "create": True,
        "delete_existing": True,
    }


async def create_downsample_dataset(
    dataset_path: str,
    start_scale: int,
    downsample_factor: list,
    cpu_cnt: int = None,
    aws_region: str = "us-west-2",
    bucket_name: str = None,
    read_cache_bytes: int = 1 << 30,
):
    """
    Create a new downsampled scale level in a multi-scale Zarr v3 dataset.

    Parameters
    ----------
    dataset_path : str
        Base S3 path of the dataset inside the bucket.
    start_scale : int
        The scale level to downsample from (e.g., 0 for original resolution).
    downsample_factor : list of int
        Downsampling factor for each of [t, c, z, y, x] dimensions.
    cpu_cnt : int, optional
        Number of threads to use. If None, uses all available CPUs.
    aws_region : str, optional
        AWS region of the S3 bucket.
    bucket_name : str, optional
        S3 bucket name.
    read_cache_bytes : int, optional
        Size of the read cache pool in bytes. Default is 1GB.

    Returns
    -------
    None
        The function writes a new downsampled scale directly to the same Zarr dataset.
    """
    if cpu_cnt is None:
        cpu_cnt = multiprocessing.cpu_count()

    kvstore_dict = {
        "driver": "file",
    }

    if bucket_name is not None:
        kvstore_dict = {
            "driver": "s3",
            "bucket": bucket_name,
            "aws_region": aws_region,
            "context": {
                "cache_pool": {"total_bytes_limit": read_cache_bytes},
                "data_copy_concurrency": {"limit": cpu_cnt},
                "s3_request_concurrency": {"limit": cpu_cnt},
                "experimental_s3_rate_limiter": {
                    "read_rate": cpu_cnt,
                    "write_rate": cpu_cnt,
                },
            },
        }

    downsample_factor = [1] * (5 - len(downsample_factor)) + downsample_factor

    source_w_down_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factor,
        "downsample_method": "mean",
        "base": {
            "driver": "zarr3",
            "kvstore": {
                **kvstore_dict,
                "path": dataset_path,
            },
            "path": str(start_scale),
            "recheck_cached_metadata": False,
            "recheck_cached_data": False,
        },
    }

    downsampled_dataset = await ts.open(spec=source_w_down_spec)
    source_dataset = downsampled_dataset.base
    new_scale = start_scale + 1

    downsampled_resolution = [
        unit.multiplier if isinstance(unit, ts.Unit) else unit
        for unit in downsampled_dataset.dimension_units
    ]

    # Creates downsample spec
    # Keeping chunksize equal in all dimensions, however it might be
    # suboptimal? We might want to have chunks of ~50-100 MB
    down_spec = create_spec(
        output_path=dataset_path,
        data_shape=downsampled_dataset.shape,
        data_dtype=str(source_dataset.dtype),
        shard_shape=source_dataset.chunk_layout.write_chunk.shape,
        chunk_shape=source_dataset.chunk_layout.read_chunk.shape,
        zyx_resolution=downsampled_resolution,
        scale=new_scale,
        cpu_cnt=cpu_cnt,
        aws_region=aws_region,
        bucket_name=bucket_name,
    )

    down_dataset = await ts.open(down_spec)
    downsampled_data = await downsampled_dataset.read()
    await down_dataset.write(downsampled_data)


async def write_tasks(list_of_tasks: List):
    """
    Gathers all the tensorstore tasks

    Parameters
    ----------
    list_of_tasks: List
        List of tensorstore tasks
    """
    # Wait for all tasks to complete
    await asyncio.gather(*list_of_tasks)


def _build_ome(
    data_shape: Tuple[int, ...],
    image_name: str,
    channel_names: Optional[List[str]] = None,
    channel_colors: Optional[List[int]] = None,
    channel_minmax: Optional[List[Tuple[float, float]]] = None,
    channel_startend: Optional[List[Tuple[float, float]]] = None,
) -> Dict:
    """
    Create the necessary metadata for an OME tiff image

    Parameters
    ----------
    data_shape: A 5-d tuple, assumed to be TCZYX order
    image_name: The name of the image
    channel_names: The names for each channel
    channel_colors: List of all channel colors
    channel_minmax: List of all (min, max) pairs of channel pixel
    ranges (min value of darkest pixel, max value of brightest)
    channel_startend: List of all pairs for rendering where start is
    a pixel value of darkness and end where a pixel value is
    saturated

    Returns
    -------
    Dict: An "omero" metadata object suitable for writing to ome-zarr
    """
    if channel_names is None:
        channel_names = [
            f"Channel:{image_name}:{i}" for i in range(data_shape[1])
        ]
    if channel_colors is None:
        channel_colors = [i for i in range(data_shape[1])]
    if channel_minmax is None:
        channel_minmax = [(0.0, 1.0) for _ in range(data_shape[1])]
    if channel_startend is None:
        channel_startend = channel_minmax

    ch = []
    for i in range(data_shape[1]):
        ch.append(
            {
                "active": True,
                "coefficient": 1,
                "color": f"{channel_colors[i]:06x}",
                "family": "linear",
                "inverted": False,
                "label": channel_names[i],
                "window": {
                    "end": float(channel_startend[i][1]),
                    "max": float(channel_minmax[i][1]),
                    "min": float(channel_minmax[i][0]),
                    "start": float(channel_startend[i][0]),
                },
            }
        )

    omero = {
        "id": 1,  # ID in OMERO
        "name": image_name,  # Name as shown in the UI
        "version": "0.4",  # Current version
        "channels": ch,
        "rdefs": {
            "defaultT": 0,  # First timepoint to show the user
            "defaultZ": data_shape[2] // 2,  # First Z section to show the user
            "model": "color",  # "color" or "greyscale"
        },
    }
    return omero


def _compute_scales(
    scale_num_levels: int,
    scale_factor: Tuple[float, float, float],
    pixelsizes: Tuple[float, float, float],
    chunks: Tuple[int, int, int, int, int],
    data_shape: Tuple[int, int, int, int, int],
    translation: Optional[List[float]] = None,
) -> Tuple[List, List]:
    """
    Generate the list of coordinate transformations
    and associated chunk options.

    Parameters
    ----------
    scale_num_levels: the number of downsampling levels
    scale_factor: a tuple of scale factors in each spatial dimension (Z, Y, X)
    pixelsizes: a list of pixel sizes in each spatial dimension (Z, Y, X)
    chunks: a 5D tuple of integers with size of each
    chunk dimension (T, C, Z, Y, X)
    data_shape: a 5D tuple of the full resolution image's shape
    translation: a 5 element list specifying the offset
    in physical units in each dimension

    Returns
    -------
    A tuple of the coordinate transforms and chunk options
    """
    transforms = [
        [
            # the voxel size for the first scale level
            {
                "type": "scale",
                "scale": [
                    1.0,
                    1.0,
                    pixelsizes[0],
                    pixelsizes[1],
                    pixelsizes[2],
                ],
            }
        ]
    ]
    if translation is not None:
        transforms[0].append(
            {"type": "translation", "translation": translation}
        )
    chunk_sizes = []
    lastz = data_shape[2]
    lasty = data_shape[3]
    lastx = data_shape[4]
    opts = dict(
        chunks=(
            1,
            1,
            min(lastz, chunks[2]),
            min(lasty, chunks[3]),
            min(lastx, chunks[4]),
        )
    )
    chunk_sizes.append(opts)
    if scale_num_levels > 1:
        for i in range(scale_num_levels - 1):
            last_transform = transforms[-1][0]
            last_scale = cast(List, last_transform["scale"])
            transforms.append(
                [
                    {
                        "type": "scale",
                        "scale": [
                            1.0,
                            1.0,
                            last_scale[2] * scale_factor[0],
                            last_scale[3] * scale_factor[1],
                            last_scale[4] * scale_factor[2],
                        ],
                    }
                ]
            )
            if translation is not None:
                transforms[-1].append(
                    {"type": "translation", "translation": translation}
                )
            lastz = int(np.ceil(lastz / scale_factor[0]))
            lasty = int(np.ceil(lasty / scale_factor[1]))
            lastx = int(np.ceil(lastx / scale_factor[2]))
            opts = dict(
                chunks=(
                    1,
                    1,
                    min(lastz, chunks[2]),
                    min(lasty, chunks[3]),
                    min(lastx, chunks[4]),
                )
            )
            chunk_sizes.append(opts)

    return transforms, chunk_sizes


def _get_axes_5d(
    time_unit: str = "millisecond", space_unit: str = "micrometer"
) -> List[Dict]:
    """Generate the list of axes.

    Parameters
    ----------
    time_unit: the time unit string, e.g., "millisecond"
    space_unit: the space unit string, e.g., "micrometer"

    Returns
    -------
    A list of dictionaries for each axis
    """
    axes_5d = [
        {"name": "t", "type": "time", "unit": f"{time_unit}"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": f"{space_unit}"},
        {"name": "y", "type": "space", "unit": f"{space_unit}"},
        {"name": "x", "type": "space", "unit": f"{space_unit}"},
    ]
    return axes_5d


def write_multiscales_metadata(
    group: zarr.Group,
    datasets: list[dict],
    fmt: Format = CurrentFormat(),
    axes: AxesType = None,
    name: str | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    """
    Write the multiscales metadata in the group.

    :type group: :class:`zarr.Group`
    :param group: The group within the zarr store to write the metadata in.
    :type datasets: list of dicts
    :param datasets:
      The list of datasets (dicts) for this multiscale image.
      Each dict must include 'path' and a 'coordinateTransformations'
      list for version 0.4 or later that must include a 'scale' transform.
    :type fmt: :class:`ome_zarr.format.Format`, optional
    :param fmt:
      The format of the ome_zarr data which should be used.
      Defaults to the most current.
    :type axes: list of str or list of dicts, optional
    :param axes:
      The names of the axes. e.g. ["t", "c", "z", "y", "x"].
      Ignored for versions 0.1 and 0.2. Required for version 0.3 or greater.
    """

    ndim = -1
    if axes is not None:
        if fmt.version in ("0.1", "0.2"):
            LOGGER.info("axes ignored for version 0.1 or 0.2")
            axes = None
        else:
            axes = _get_valid_axes(axes=axes, fmt=fmt)
            if axes is not None:
                ndim = len(axes)
    if (
        isinstance(metadata, dict)
        and metadata.get("metadata")
        and isinstance(metadata["metadata"], dict)
        and "omero" in metadata["metadata"]
    ):
        omero_metadata = metadata["metadata"].get("omero")
        if omero_metadata is None:
            raise KeyError("If `'omero'` is present, value cannot be `None`.")
        for c in omero_metadata["channels"]:
            if "color" in c:
                if not isinstance(c["color"], str) or len(c["color"]) != 6:
                    raise TypeError("`'color'` must be a hex code string.")
            if "window" in c:
                if not isinstance(c["window"], dict):
                    raise TypeError("`'window'` must be a dict.")
                for p in ["min", "max", "start", "end"]:
                    if p not in c["window"]:
                        raise KeyError(f"`'{p}'` not found in `'window'`.")
                    if not isinstance(c["window"][p], (int, float)):
                        raise TypeError(f"`'{p}'` must be an int or float.")

        group.attrs["omero"] = omero_metadata

    # note: we construct the multiscale metadata via dict(), rather than {}
    # to avoid duplication of protected keys like 'version' in **metadata
    # (for {} this would silently over-write it, with dict() it explicitly fails)
    multiscales = [
        dict(
            version=fmt.version,
            datasets=_validate_datasets(datasets, ndim, fmt),
            name=name or group.name,
            **metadata,
        )
    ]
    if axes is not None:
        multiscales[0]["axes"] = axes

    group.attrs["multiscales"] = multiscales


def write_ome_ngff_metadata(
    group: zarr.Group,
    arr_shape: List[int],
    chunksize: List[int],
    image_name: str,
    n_lvls: int,
    scale_factors: tuple,
    voxel_size: tuple,
    channel_names: List[str] = None,
    channel_colors: List[str] = None,
    channel_minmax: List[float] = None,
    channel_startend: List[float] = None,
    metadata: dict = None,
):
    """
    Write OME-NGFF metadata to a Zarr group.

    Parameters
    ----------
    group : zarr.Group
        The output Zarr group.
    arr_shape : List[int]
        List of ints with the dataset shape.
    image_name : str
        The name of the image.
    n_lvls : int
        The number of pyramid levels.
    scale_factors : tuple
        The scale factors for downsampling along each dimension.
    voxel_size : tuple
        The voxel size along each dimension.
    channel_names: List[str]
        List of channel names to add to the OMENGFF metadata
    channel_colors: List[str]
        List of channel colors to visualize the data
    chanel_minmax: List[float]
        List of channel min and max values based on the
        image dtype
    channel_startend: List[float]
        List of the channel start and end metadata. This is
        used for visualization. The start and end range might be
        different from the min max and it is usually inside the
        range
    metadata: dict
        Extra metadata to write in the OME-NGFF metadata
    """
    if metadata is None:
        metadata = {}
    fmt = CurrentFormat()

    # Building the OMERO metadata
    ome_json = _build_ome(
        arr_shape,
        image_name,
        channel_names=channel_names,
        channel_colors=channel_colors,
        channel_minmax=channel_minmax,
        channel_startend=channel_startend,
    )
    group.attrs["omero"] = ome_json
    axes_5d = _get_axes_5d()
    coordinate_transformations, chunk_opts = _compute_scales(
        n_lvls, scale_factors, voxel_size, chunksize, arr_shape, None
    )
    fmt.validate_coordinate_transformations(
        len(arr_shape), n_lvls, coordinate_transformations
    )
    # Setting coordinate transfomations
    datasets = [{"path": str(i)} for i in range(n_lvls)]
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    # Writing the multiscale metadata
    write_multiscales_metadata(group, datasets, fmt, axes_5d, **metadata)


def _get_pyramid_metadata():
    """
    Gets the image pyramid metadata
    using tensorstore package
    """
    return {
        "metadata": {
            "description": "Downscaling tensorstore downsample",
            "method": "tensorstore.downsample",
            "version": "0.1.72",
            "args": "[false]",
            # No extra parameters were used different
            # from the orig. array and scales
            "kwargs": {},
        }
    }


def czi_stack_zarr_writer(
    czi_path: str,
    output_path: str,
    voxel_size: List[float],
    shardsize: List[int],
    chunksize: List[int],
    scale_factor: List[int],
    n_lvls: int,
    channel_name: str,
    logger: logging.Logger,
    stack_name: str,
):
    """
    Writes a fused Zeiss channel in OMEZarr
    format. This channel was read as a lazy array.

    Parameters
    ----------
    czi_path: str
        Path where the CZI file is stored.

    output_path: PathLike
        Path where we want to write the OMEZarr
        channel

    voxel_size: List[float]
        Voxel size representing the dataset

    chunksize: List[int]
        Final chunksize we want to use to write
        the final dataset

    codec: str
        Image codec for writing the Zarr

    compression_level: int
        Compression level

    scale_factor: List[int]
        Scale factor per axis. The dimensionality
        is organized as ZYX.

    n_lvls: int
        Number of levels on the pyramid (multiresolution)
        for better visualization

    channel_name: str
        Channel name we are currently writing

    logger: logging.Logger
        Logger object

    """
    written_pyramid = []
    start_time = time.time()

    with czifile.CziFile(str(czi_path)) as czi:
        dataset_shape = tuple(i for i in czi.shape if i != 1)
        extra_axes = (1,) * (5 - len(dataset_shape))
        dataset_shape = extra_axes + dataset_shape

        shardsize = ([1] * (5 - len(chunksize))) + chunksize
        chunksize = ([1] * (5 - len(chunksize))) + chunksize

        # Getting channel color
        channel_colors = None

        print(f"Writing {dataset_shape} from {stack_name} to {output_path}")

        if np.issubdtype(czi.dtype, np.integer):
            np_info_func = np.iinfo

        else:
            # Floating point
            np_info_func = np.finfo

        # Getting min max metadata for the dtype
        channel_minmax = [
            (
                np_info_func(czi.dtype).min,
                np_info_func(czi.dtype).max,
            )
            for _ in range(dataset_shape[1])
        ]

        # Setting values for CZI
        # Ideally we would use da.percentile(image_data, (0.1, 95))
        # However, it would take so much time and resources and it is
        # not used that much on neuroglancer
        channel_startend = [(0.0, 550.0) for _ in range(dataset_shape[1])]

        # Writing OME-NGFF metadata
        write_ome_ngff_metadata(
            group=new_channel_group,
            arr_shape=dataset_shape,
            image_name=stack_name,
            n_lvls=n_lvls,
            scale_factors=scale_factor,
            voxel_size=voxel_size,
            channel_names=[channel_name],
            channel_colors=channel_colors,
            channel_minmax=channel_minmax,
            channel_startend=channel_startend,
            metadata=_get_pyramid_metadata(),
            chunksize=chunksize,
        )

        # Full resolution spec
        spec = create_spec(
            output_path=output_path,
            data_shape=dataset_shape,
            data_dtype=czi.dtype.name,
            shard_shape=shardsize,
            chunk_shape=chunksize,
            zyx_resolution=voxel_size,
            compressor_kwargs=compressor_kwargs,
        )

        tasks = []
        dataset = ts.open(spec).result()

        # chunksize must be TCZYX order
        for block, axis_area in czi_block_generator(
            czi,
            axis_jumps=chunksize[-3],
            slice_axis="z",
        ):
            region = (
                slice(None),
                slice(None),
                axis_area,
                slice(0, dataset_shape[-2]),
                slice(0, dataset_shape[-1]),
            )
            write_task = dataset[region].write(pad_array_n_d(block))
            tasks.append(write_task)

        # Waiting for the tensorstore tasks
        asyncio.run(write_tasks(tasks))

        for level in range(n_lvls):
            print(f"Writing scale {level+1}")
            asyncio.run(
                create_downsample_dataset(
                    dataset_path=output_path,
                    start_scale=level,
                    downsample_factor=scale_factor,
                )
            )

    end_time = time.time()
    logger.info(f"Time to write the dataset: {end_time - start_time}")
    print(f"Time to write the dataset: {end_time - start_time}")
    logger.info(f"Written pyramid: {written_pyramid}")


def example():
    """
    Conversion example
    """
    import time
    from pathlib import Path

    czi_test_stack = Path(
        "/Users/camilo.laiton/repositories/Z1/czi_to_zarr/data/tiles_test/SPIM/488_large.czi"
    )

    if czi_test_stack.exists():
        writing_opts = create_czi_opts(codec="zstd", compression_level=3)
        start_time = time.time()

        # for channel_name in
        # for i, chn_name in enumerate(czi_file_reader.channel_names):
        czi_stack_zarr_writer(
            czi_path=str(czi_test_stack),
            output_path=f"./{czi_test_stack.stem}",
            voxel_size=[1.0, 1.0, 1.0],
            chunksize=[128, 128, 128],
            scale_factor=[2, 2, 2],
            n_lvls=4,
            channel_name=czi_test_stack.stem,
            logger=logging.Logger(name="test"),
            stack_name="test_conversion_czi_package.zarr",
            writing_options=writing_opts["compressor"],
        )
        end_time = time.time()
        print(f"Conversion time: {end_time - start_time} s")

    else:
        print(f"File does not exist: {czi_test_stack}")


if __name__ == "__main__":
    example()
