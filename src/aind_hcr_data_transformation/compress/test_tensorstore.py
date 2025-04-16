import asyncio
import configparser
import json
import multiprocessing
import os

import boto3
import numpy as np
import tensorstore as ts
import zarr


def get_aws_credentials(profile="default"):
    config = configparser.ConfigParser()
    config.read(os.path.expanduser("~/.aws/credentials"))

    if profile in config:
        return {
            "access_key_id": config[profile]["aws_access_key_id"],
            "secret_access_key": config[profile]["aws_secret_access_key"],
        }
    else:
        return None


async def write_zarr_v3(data, spec):
    # Open the dataset for writing
    dataset = await ts.open(spec)
    print(f"Dataset info: {dataset}")
    print(f"TensorStore dataset created with shape: {dataset.shape}")

    # Write the data
    await dataset.write(data)
    print("Data written successfully")

    # Read back a portion to verify
    subset = await dataset[10:20, 30:40, 50:60].read()
    print(f"Read subset shape: {subset.shape}")
    print(
        f"Subset values match: {np.allclose(subset, data[10:20, 30:40, 50:60])}"
    )

    # Print out metadata
    # In Zarr v3, metadata is stored in the zarr.json file
    with open("./my_zarr_v3_dataset/zarr.json", "r") as f:
        zarr_metadata = json.load(f)
    print("Zarr v3 metadata:")
    print(json.dumps(zarr_metadata, indent=2))


# Additional example: Read the dataset
async def read_zarr_v3(data):
    # Open the existing dataset for reading
    read_spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": "my_zarr_v3_dataset"},
    }

    dataset = await ts.open(read_spec)
    print(f"Opened existing dataset with shape: {dataset.shape}")

    # Read all data
    read_data = await dataset.read()
    print(f"Read data shape: {read_data.shape}")
    print(f"Read dataset: {dataset}")
    print(f"All values match: {np.allclose(read_data, data)}")
    print(f"Read data: {read_data}")

    return read_data


def create_spec(
    output_path,
    data_shape,
    data_dtype,
    shard_shape,
    chunk_shape,
    zyx_resolution,
    scale="0",
    cpu_cnt=None,
    aws_region="us-west-2",
):
    if cpu_cnt == None:
        cpu_cnt = multiprocessing.cpu_count()

    zyx_resolution = [f"{r}um" for r in zyx_resolution if r is not None]
    zyx_resolution = [None] * (5 - len(zyx_resolution)) + zyx_resolution

    spec = {
        "driver": "zarr3",
        "kvstore": {
            # "driver": "file",
            "driver": "s3",
            "bucket": "aind-msma-morphology-data",
            "path": output_path,
            "aws_region": aws_region,
            "aws_credentials": {
                "type": "profile",
                "profile": "default",
                "credentials_file": os.path.expanduser("~/.aws/credentials"),
            },
            "context": {
                "cache_pool": {
                    "total_bytes_limit": 1 << 30
                },  # 1 GB read cache
                "data_copy_concurrency": {"limit": cpu_cnt},
                "s3_request_concurrency": {"limit": cpu_cnt},
                "experimental_s3_rate_limiter": {
                    "read_rate": cpu_cnt,
                    "write_rate": cpu_cnt,
                },
            },
        },
        "path": str(scale),
        # Disables recheck in metadata when i/o in chunks
        "recheck_cached_metadata": False,
        "recheck_cached_data": False,
        "metadata": {
            "shape": data_shape,
            "zarr_format": 3,
            "node_type": "array",
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": shard_shape,
                },
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {
                    "separator": "/",
                },
            },
            "attributes": {
                "dimension_units": zyx_resolution,
            },
            "dimension_names": [
                "t",
                "c",
                "z",
                "y",
                "x",
            ],  # Optional but helpful
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": chunk_shape,
                        "codecs": [
                            {
                                "name": "bytes",
                                "configuration": {
                                    "endian": "little",
                                },
                            },
                            {
                                "name": "blosc",
                                "configuration": {
                                    "cname": "zstd",
                                    "clevel": 3,
                                    "shuffle": "shuffle",
                                },
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

    return spec


async def write_tasks(list_of_tasks):
    # Wait for all tasks to complete
    await asyncio.gather(*list_of_tasks)


async def create_downsample_dataset_2(
    dataset_path, start_scale, downsample_factor, cpu_cnt=None
):
    if cpu_cnt == None:
        cpu_cnt = multiprocessing.cpu_count()

    downsample_factor = [1] * (5 - len(downsample_factor)) + downsample_factor

    # Getting source dataset with downsample spec
    source_w_down_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factor,
        "downsample_method": "mean",
        "base": {
            "driver": "zarr3",
            "kvstore": {
                # "driver": "file",
                "driver": "s3",
                "bucket": "aind-msma-morphology-data",
                "path": dataset_path,
                "context": {
                    "cache_pool": {"total_bytes_limit": 1 << 30},
                    "cache_pool#remote": {"total_bytes_limit": 1 << 30},
                    "data_copy_concurrency": {"limit": cpu_cnt},
                    "s3_request_concurrency": {"limit": cpu_cnt},
                    "experimental_s3_rate_limiter": {
                        "write_rate": cpu_cnt,
                        "read_rate": cpu_cnt,
                    },
                },
            },
            "path": str(start_scale),
            # Disables recheck in metadata when i/o in chunks
            "recheck_cached_metadata": False,
            "recheck_cached_data": False,
        },
    }

    downsampled_dataset = await ts.open(
        spec=source_w_down_spec,
    )
    source_dataset = downsampled_dataset.base
    print("Source dataset ", source_dataset)
    # z = zarr.open(f"{dataset_path}")
    # print(list(z.keys()))
    # print(z[str(start_scale)].shape)
    new_scale = start_scale + 1

    # Creating new downsampled resolution based on scale factors
    downsampled_resolution = [
        du.multiplier if isinstance(du, ts.Unit) else du
        for du in downsampled_dataset.dimension_units
    ]
    # print(downsampled_resolution)

    # Creating downsampled spec
    down_spec = create_spec(
        output_path=dataset_path,
        data_shape=downsampled_dataset.shape,
        data_dtype=source_dataset.dtype,
        shard_shape=source_dataset.chunk_layout.write_chunk.shape,
        # source_dataset.chunk_layout.write_chunk.shape,
        chunk_shape=source_dataset.chunk_layout.read_chunk.shape,
        # source_dataset.chunk_layout.read_chunk.shape,
        zyx_resolution=downsampled_resolution,
        scale=new_scale,
    )
    # print("Downsampled spec: ", down_spec)
    down_dataset = await ts.open(down_spec)
    # print(f"Created level {new_scale} dataset with shape: {down_dataset.shape}")

    # Reading data
    downsampled_data = await downsampled_dataset.read()
    # print(f"Read downsampled data with shape: {downsampled_data.shape}")

    # Writing data
    await down_dataset.write(downsampled_data)
    # print(f"Written level {new_scale} data")


def get_s3_dir_size(bucket_name, prefix):
    s3 = boto3.client("s3")
    total_size = 0
    continuation_token = None

    while True:
        if continuation_token:
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix,
                ContinuationToken=continuation_token,
            )
        else:
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

        for obj in response.get("Contents", []):
            total_size += obj["Size"]

        if response.get("IsTruncated"):  # More data to fetch
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return total_size


def main():
    import time
    from pathlib import Path

    import czi_to_zarr
    import czifile

    # Execute the async function
    # asyncio.run(write_zarr_v3())
    # asyncio.run(read_zarr_v3())

    BASE_PATH = (
        "/Users/camilo.laiton/repositories/Z1/czi_to_zarr/data/tiles_test/SPIM"
    )
    czi_test_stack = Path(f"{BASE_PATH}/488_large.czi")
    multiscale_levels = 5
    output_path = "test_data/SmartSPIM/"

    test_chunksize = [128]  # , 128, 64]
    test_shardsize = [512]  # , 256, 128]

    for shard_size in test_shardsize:
        curr_shard_size = [1, 1] + [shard_size] * 3

        for chunk_size in test_chunksize:

            if chunk_size >= shard_size:
                print(
                    f"Skipping benchmark for chunksize {chunk_size} because shard is: {shard_size}"
                )
                continue

            curr_output_path = f"{output_path}/zarr_v3test_sh_{shard_size}_ch_{chunk_size}.zarr"
            curr_chunk_size = [1, 1] + [chunk_size] * 3

            if czi_test_stack.exists():

                print(
                    f"*** Running Shard {shard_size} - Chunk {chunk_size} ***"
                )
                print(f"Output path: {curr_output_path}")
                start_time = time.time()
                with czifile.CziFile(str(czi_test_stack)) as czi:
                    dataset_shape = tuple(i for i in czi.shape if i != 1)
                    extra_axes = (1,) * (5 - len(dataset_shape))
                    dataset_shape = extra_axes + dataset_shape

                    spec = create_spec(
                        output_path=curr_output_path,
                        data_shape=dataset_shape,
                        data_dtype="uint16",
                        shard_shape=curr_shard_size,
                        chunk_shape=curr_chunk_size,
                        zyx_resolution=[1.0, 1.0, 1.0],
                    )

                    print(f"Writing {dataset_shape}: spec: {spec}")
                    dataset = ts.open(spec).result()
                    print(
                        f"Dataset info: {dataset} - Layout: {dataset.chunk_layout}"
                    )

                    tasks = []
                    for block, axis_area in czi_to_zarr.czi_block_generator(
                        czi,
                        axis_jumps=curr_shard_size[-3],
                        slice_axis="z",
                    ):
                        region = (
                            slice(None),
                            slice(None),
                            axis_area,
                            slice(0, dataset_shape[-2]),
                            slice(0, dataset_shape[-1]),
                        )
                        print(f"Writing in region {region}")

                        # dataset[region].write(
                        #     czi_to_zarr.pad_array_n_d(block)
                        # ).result()
                        block = czi_to_zarr.pad_array_n_d(block)
                        write_task = dataset[region].write(block)
                        tasks.append(write_task)

                print("Starting asyncio!")
                asyncio.run(write_tasks(tasks))
                full_res_time = time.time()

                times = []
                print("Starting multiscale!")
                for level in range(multiscale_levels):
                    level_start_time = time.time()
                    print(f"Writing scale {level+1}")
                    asyncio.run(
                        create_downsample_dataset_2(
                            dataset_path=curr_output_path,
                            start_scale=level,
                            downsample_factor=[2, 2, 2],
                        )
                    )
                    level_end_time = time.time()
                    times.append(level_end_time - level_start_time)

                elapsed_time = time.time() - start_time
                total_bytes = get_s3_dir_size(
                    bucket_name="aind-msma-morphology-data",
                    prefix=curr_output_path,
                )
                total_gb = total_bytes / (1024**3)

                print(f"\n✅ All blocks written successfully")
                print(f"🕒 Time taken: {elapsed_time:.2f} seconds")
                print(
                    f"🕒 Time taken for res 0: {full_res_time-start_time:.2f} seconds"
                )

                for i, t in enumerate(times):
                    print(f"🕒 Time taken for res {i+1}: {t:.2f} seconds")

                print(
                    f"💾 Total size written: {total_gb:.2f} GB ({total_bytes:,} bytes)"
                )

                # # First, let's create some sample data
                # data = np.random.random((256, 256, 256)).astype(np.float32)
                # print(data)
                # print(f"Original data shape: {data.shape}, dtype: {data.dtype}")

                # # Define the Zarr v3 store specification
                # # Note: TensorStore supports Zarr v3 which is identified by the "zarr3" driver

                # asyncio.run(write_zarr_v3(data, spec))
                # asyncio.run(read_zarr_v3(data))

            else:
                print(f"File does not exist: {czi_test_stack}")


if __name__ == "__main__":
    main()
