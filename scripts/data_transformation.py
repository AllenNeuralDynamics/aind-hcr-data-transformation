"""
Run data transformation locally
"""

import sys

from aind_protein_data_transformation.models import ZeissJobSettings
from aind_protein_data_transformation.zeiss_job import ZeissCompressionJob
from typing import Optional
from pathlib import Path
from time import time
from aind_protein_data_transformation.compress.czi_to_zarr import (
    czi_stack_zarr_writer,
)
from aind_protein_data_transformation.utils import utils
from bioio import BioImage
import bioio_czi
import dask.array as da
import logging
import shutil

def transform_pan_protein_data(
    input_source: str,
    output_dir: str,
    s3_location: str,
    num_partitions: Optional[int] = 1,
    remove_stacks: Optional[bool] = False
):
    """
    Transforms pan protein data.

    Parameters
    ----------
    input_source: str
        Path where the CZI files are stored.

    output_dir: str
        Path where we want to write the zarrs.
    
    num_partitions: Optional[int]
        Number of partitions. Default: 1
    """
    basic_job_settings = ZeissJobSettings(
        input_source=input_source,
        output_directory=output_dir,
        num_of_partitions=num_partitions,
        partition_to_process=0,
        s3_location=s3_location,
    )
    basic_job = ZeissCompressionJob(job_settings=basic_job_settings)
    
    # Commenting it out since there are datasets which could have
    # multiple channels, I prefer to have each zarr as a channel
    # basic_job.run_job()
    
    compressor = basic_job._get_compressor()
    
    job_start_time = time()

    partitioned_list = basic_job._get_partitioned_list_of_stack_paths()
    stacks_to_process = partitioned_list[
        basic_job.job_settings.partition_to_process
    ]
    
    for stack in stacks_to_process:
        print(f"Converting {stack}")
        stack_name = stack.stem
        root_name = stack.parent.stem

        output_path = Path(basic_job.job_settings.output_directory).joinpath(stack_name)

        czi_file_reader = BioImage(str(stack), reader=bioio_czi.Reader)

        voxel_size_zyx = czi_file_reader.physical_pixel_sizes
        voxel_size_zyx = [
            voxel_size_zyx.Z,
            voxel_size_zyx.Y,
            voxel_size_zyx.X,
        ]
        delayed_stack = da.squeeze(czi_file_reader.dask_data)

        msg = (
            f"Voxel resolution ZYX {voxel_size_zyx} "
            f"with name {stack_name} - {delayed_stack} - output: {output_path}"
        )
        print(msg)
        C, Z, H, W = delayed_stack.shape

        for idx in range(C):
            print(f"Processing {stack_name} - channel {idx}")
            curr_stack_name = f"channel_{idx}"
            czi_stack_zarr_writer(
                image_data=delayed_stack[idx],
                output_path=output_path,
                voxel_size=voxel_size_zyx,
                final_chunksize=basic_job.job_settings.chunk_size,
                scale_factor=basic_job.job_settings.scale_factor,
                n_lvls=basic_job.job_settings.downsample_levels,
                channel_name=curr_stack_name,
                stack_name=f"{curr_stack_name}.ome.zarr",
                logger=logging,
                writing_options=compressor,
            )

            if basic_job.job_settings.s3_location is not None:
                channel_zgroup_file = output_path / ".zgroup"
                s3_channel_zgroup_file = (
                    f"{basic_job.job_settings.s3_location}/{stack_name}/.zgroup"
                )
                print(
                    f"Uploading {channel_zgroup_file} to "
                    f"{s3_channel_zgroup_file}"
                )
                utils.copy_file_to_s3(
                    channel_zgroup_file, s3_channel_zgroup_file
                )
                ome_zarr_curr_stack_name = f"{curr_stack_name}.ome.zarr"
                ome_zarr_stack_path = output_path.joinpath(ome_zarr_curr_stack_name)
                s3_stack_dir = (
                    f"{basic_job.job_settings.s3_location}/{stack_name}/"
                    f"{ome_zarr_curr_stack_name}"
                )
                print(
                    f"Uploading {ome_zarr_stack_path} to {s3_stack_dir}"
                )
                utils.sync_dir_to_s3(ome_zarr_stack_path, s3_stack_dir)
                print(f"Removing: {ome_zarr_stack_path}")
                # Remove stack if uploaded to s3. We can potentially do all
                # the stacks in the partition in parallel using dask to speed
                # this up
                if remove_stacks:
                    shutil.rmtree(ome_zarr_stack_path)

    total_job_duration = time() - job_start_time
    print(f"Total job duration: {total_job_duration}")

if __name__ == "__main__":
    BASE_PATH = Path("/allen/aind/scratch/camilo.laiton/pan_protein_data_deep_learning")
    input_source = BASE_PATH.joinpath("CZI_data")
    output_dir = BASE_PATH.joinpath("OMEZarr_data")

    if input_source.exists():
        print(f"Starting transformation in {input_source}")
        transform_pan_protein_data(
            input_source=str(input_source),
            output_dir=str(output_dir),
            num_partitions=1,
            s3_location="s3://aind-msma-morphology-data/test_data/protein_project"
        )