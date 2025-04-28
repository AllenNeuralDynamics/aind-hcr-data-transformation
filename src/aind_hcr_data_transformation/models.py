"""Helpful models used in the compression job"""

from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, Literal

from aind_data_transformation.core import BasicJobSettings
from numcodecs import Blosc
from pydantic import Field

PathLike = Union[str, Path]


class CompressorName(str, Enum):
    """Enum for compression algorithms a user can select"""

    BLOSC = Blosc.codec_id


class ZeissJobSettings(BasicJobSettings):
    """ZeissCompressionJob settings."""

    input_source: PathLike = Field(
        ...,
        description=("Source of the Zeiss stack data."),
    )
    output_directory: PathLike = Field(
        ...,
        description=("Where to write the data to locally."),
    )
    s3_location: Optional[str] = None
    num_of_partitions: int = Field(
        ...,
        description=(
            "This script will generate a list of individual stacks, "
            "and then partition the list into this number of partitions."
        ),
    )
    partition_to_process: int = Field(
        ...,
        description=("Which partition of stacks to process. "),
    )
    compressor_name: CompressorName = Field(
        default=CompressorName.BLOSC,
        description="Type of compressor to use.",
        title="Compressor Name.",
    )
    # It will be safer if these kwargs fields were objects with known schemas
    compressor_kwargs: dict = Field(
        default={"cname": "zstd", "clevel": 3, "shuffle": "shuffle"},
        description="Arguments to be used for the compressor.",
        title="Compressor Kwargs",
    )
    compress_job_save_kwargs: dict = Field(
        default={"n_jobs": -1},  # -1 to use all available cpu cores.
        description="Arguments for recording save method.",
        title="Compress Job Save Kwargs",
    )
    shard_size: List[int] = Field(
        default=[512, 512, 512],  # Default list with three integers
        description="Shard size in axis, a list of three integers",
        title="Shard Size",
    )
    chunk_size: List[int] = Field(
        default=[128, 128, 128],  # Default list with three integers
        description="Chunk size in axis, a list of three integers",
        title="Chunk Size",
    )
    scale_factor: List[int] = Field(
        default=[2, 2, 2],  # Default list with three integers
        description="Scale factors in axis, a list of three integers",
        title="Scale Factors",
    )
    downsample_levels: int = Field(
        default=5,
        description="The number of levels of the image pyramid",
        title="Downsample Levels",
    )
    downsample_mode: Literal["stride", "median", "mode", "mean", "min", "max"] = Field(
        default="mean",
        description="Downsample mode",
        title="Downsample Mode",
    )
    tensorstore_batch_size: int = Field(
        default=3,
        description="Batch size to execute concurrent tensorstore tasks",
        title="Tensorstore batch size",
    )
