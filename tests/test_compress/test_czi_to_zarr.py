"""Test suite for the ZarrV3 data transformation module."""

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np

from aind_hcr_data_transformation.compress.czi_to_zarr import (
    create_downsample_dataset,
    create_spec,
    czi_stack_zarr_writer,
)


class TestCreateSpec(unittest.TestCase):
    """Test cases for the create_spec function."""

    def test_create_spec_basic_parameters(self):
        """Test create_spec with basic parameters."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="uint16",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[1.0, 0.5, 0.5],
            compressor_kwargs={"cname": "zstd", "clevel": 5},
        )

        self.assertEqual(spec["driver"], "zarr3")
        self.assertEqual(spec["kvstore"]["driver"], "file")
        self.assertEqual(spec["kvstore"]["path"], "/test/path")
        self.assertEqual(spec["path"], "0")  # default scale
        self.assertEqual(spec["metadata"]["shape"], [1, 1, 100, 200, 300])
        self.assertEqual(spec["metadata"]["data_type"], "uint16")
        self.assertEqual(
            spec["metadata"]["chunk_grid"]["configuration"]["chunk_shape"],
            [1, 1, 50, 100, 150],
        )

        # Check sharding codec configuration
        sharding_codec = spec["metadata"]["codecs"][0]
        self.assertEqual(sharding_codec["name"], "sharding_indexed")
        self.assertEqual(
            sharding_codec["configuration"]["chunk_shape"], [1, 1, 25, 50, 75]
        )

        # Check compressor is blosc with provided kwargs
        blosc_codec = sharding_codec["configuration"]["codecs"][1]
        self.assertEqual(blosc_codec["name"], "blosc")
        self.assertEqual(
            blosc_codec["configuration"], {"cname": "zstd", "clevel": 5}
        )

    def test_create_spec_with_s3_bucket(self):
        """Test create_spec with S3 bucket configuration."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="float32",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[1.0, 0.5, 0.5],
            compressor_kwargs={"cname": "lz4", "clevel": 3},
            bucket_name="test-bucket",
            aws_region="us-east-1",
            cpu_cnt=8,
            read_cache_bytes=2**31,
        )

        self.assertEqual(spec["kvstore"]["driver"], "s3")
        self.assertEqual(spec["kvstore"]["bucket"], "test-bucket")
        self.assertEqual(spec["kvstore"]["aws_region"], "us-east-1")

        # Check context settings
        context = spec["kvstore"]["context"]
        self.assertEqual(context["cache_pool"]["total_bytes_limit"], 2**31)
        self.assertEqual(context["data_copy_concurrency"]["limit"], 8)
        self.assertEqual(context["s3_request_concurrency"]["limit"], 8)
        self.assertEqual(
            context["experimental_s3_rate_limiter"]["read_rate"], 8
        )
        self.assertEqual(
            context["experimental_s3_rate_limiter"]["write_rate"], 8
        )

    @patch("multiprocessing.cpu_count")
    def test_create_spec_default_cpu_count(self, mock_cpu_count):
        """Test that cpu_count defaults to multiprocessing.cpu_count()."""
        mock_cpu_count.return_value = 16

        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="uint16",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[1.0, 0.5, 0.5],
            compressor_kwargs={"cname": "zstd", "clevel": 5},
            bucket_name="test-bucket",
        )

        context = spec["kvstore"]["context"]
        self.assertEqual(context["data_copy_concurrency"]["limit"], 16)
        mock_cpu_count.assert_called_once()

    def test_create_spec_resolution_formatting(self):
        """Test that zyx_resolution is properly formatted with units."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="uint16",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[2.0, 1.0, 1.0],
            compressor_kwargs={"cname": "zstd", "clevel": 5},
        )

        expected_units = [None, None, "2.0um", "1.0um", "1.0um"]
        self.assertEqual(
            spec["metadata"]["attributes"]["dimension_units"], expected_units
        )

    def test_create_spec_with_none_resolution(self):
        """Test handling of None values in resolution."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="uint16",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[None, 0.5, 0.5],
            compressor_kwargs={"cname": "zstd", "clevel": 5},
        )

        expected_units = [None, None, None, "0.5um", "0.5um"]
        self.assertEqual(
            spec["metadata"]["attributes"]["dimension_units"], expected_units
        )

    def test_create_spec_custom_scale(self):
        """Test create_spec with custom scale parameter."""
        spec = create_spec(
            output_path="/test/path",
            data_shape=[1, 1, 100, 200, 300],
            data_dtype="uint16",
            shard_shape=[1, 1, 50, 100, 150],
            chunk_shape=[1, 1, 25, 50, 75],
            zyx_resolution=[1.0, 0.5, 0.5],
            compressor_kwargs={"cname": "zstd", "clevel": 5},
            scale="2",
        )

        self.assertEqual(spec["path"], "2")


class TestCreateDownsampleDataset(unittest.IsolatedAsyncioTestCase):
    """Test cases for the create_downsample_dataset function."""

    @patch("aind_hcr_data_transformation.compress.czi_to_zarr.create_spec")
    @patch("tensorstore.open", new_callable=AsyncMock)
    async def test_create_downsample_dataset_basic(
        self, mock_ts_open, mock_create_spec
    ):
        """Test basic downsampling functionality."""

        mock_source_dataset = Mock()
        mock_source_dataset.dtype.name = "uint16"
        mock_source_dataset.chunk_layout.write_chunk.shape = [
            1,
            1,
            50,
            100,
            150,
        ]
        mock_source_dataset.chunk_layout.read_chunk.shape = [1, 1, 25, 50, 75]

        mock_downsampled_data = np.zeros((1, 1, 50, 100, 150), dtype=np.uint16)

        # Create a mock for the downsampled dataset
        # (returned on first tensorstore.open call)
        mock_downsampled_dataset = AsyncMock()
        mock_downsampled_dataset.base = mock_source_dataset
        mock_downsampled_dataset.shape = [1, 1, 50, 100, 150]
        mock_downsampled_dataset.dimension_units = [
            None,
            None,
            "2.0um",
            "1.0um",
            "1.0um",
        ]
        mock_downsampled_dataset.read = AsyncMock(
            return_value=mock_downsampled_data
        )

        mock_output_dataset = AsyncMock()
        mock_output_dataset.write = AsyncMock(return_value=None)

        mock_ts_open.side_effect = [
            mock_downsampled_dataset,
            mock_output_dataset,
        ]

        mock_create_spec.return_value = {"driver": "zarr"}

        await create_downsample_dataset(
            dataset_path="/test/dataset",
            start_scale=0,
            downsample_factor=[2, 2, 2],
            downsample_mode="mean",
            compressor_kwargs={"cname": "zstd", "clevel": 5},
        )

        self.assertEqual(mock_ts_open.call_count, 2)

        # Check spec of first call (downsample)
        downsample_spec_call = mock_ts_open.call_args_list[0][1]["spec"]
        self.assertEqual(downsample_spec_call["driver"], "downsample")
        self.assertEqual(
            downsample_spec_call["downsample_factors"], [1, 1, 2, 2, 2]
        )
        self.assertEqual(downsample_spec_call["downsample_method"], "mean")

        # Check create_spec was called with expected arguments
        mock_create_spec.assert_called_once()
        create_spec_args = mock_create_spec.call_args.kwargs
        self.assertEqual(create_spec_args["output_path"], "/test/dataset")
        self.assertEqual(create_spec_args["scale"], 1)  # start_scale + 1

        # Ensure data was read and written
        mock_downsampled_dataset.read.assert_awaited_once()
        mock_output_dataset.write.assert_awaited_once_with(
            mock_downsampled_data
        )

    @patch("multiprocessing.cpu_count", return_value=8)
    @patch("aind_hcr_data_transformation.compress.czi_to_zarr.create_spec")
    @patch("tensorstore.open", new_callable=AsyncMock)
    async def test_create_downsample_dataset_with_s3(
        self, mock_ts_open, mock_create_spec, mock_cpu_count
    ):
        """Test downsampling with S3 configuration."""

        # Set up the source dataset
        mock_source_dataset = Mock()
        mock_source_dataset.dtype.name = "float32"
        mock_source_dataset.chunk_layout.write_chunk.shape = [1, 1, 32, 64, 64]
        mock_source_dataset.chunk_layout.read_chunk.shape = [1, 1, 16, 32, 32]

        mock_downsampled_data = np.zeros((1, 1, 25, 50, 50), dtype=np.float32)

        mock_downsampled_dataset = AsyncMock()
        mock_downsampled_dataset.base = mock_source_dataset
        mock_downsampled_dataset.shape = [1, 1, 25, 50, 50]
        mock_downsampled_dataset.dimension_units = [
            None,
            None,
            "4.0um",
            "2.0um",
            "2.0um",
        ]
        mock_downsampled_dataset.read = AsyncMock(
            return_value=mock_downsampled_data
        )

        mock_output_dataset = AsyncMock()
        mock_output_dataset.write = AsyncMock(return_value=None)

        mock_ts_open.side_effect = [
            mock_downsampled_dataset,
            mock_output_dataset,
        ]

        mock_create_spec.return_value = {"driver": "zarr"}

        await create_downsample_dataset(
            dataset_path="/test/dataset",
            start_scale=1,
            downsample_factor=[2, 2, 2],
            downsample_mode="median",
            compressor_kwargs={"cname": "lz4"},
            bucket_name="test-bucket",
            aws_region="eu-west-1",
            read_cache_bytes=2**32,
        )

        self.assertEqual(mock_ts_open.call_count, 2)

        downsample_spec = mock_ts_open.call_args_list[0][1]["spec"]
        base_kvstore = downsample_spec["base"]["kvstore"]
        self.assertEqual(base_kvstore["driver"], "s3")
        self.assertEqual(base_kvstore["bucket"], "test-bucket")
        self.assertEqual(base_kvstore["aws_region"], "eu-west-1")
        self.assertEqual(
            base_kvstore["context"]["cache_pool"]["total_bytes_limit"], 2**32
        )

        mock_downsampled_dataset.read.assert_awaited_once()
        mock_output_dataset.write.assert_awaited_once_with(
            mock_downsampled_data
        )

        mock_create_spec.assert_called_once()
        create_spec_args = mock_create_spec.call_args.kwargs
        self.assertEqual(create_spec_args["output_path"], "/test/dataset")
        self.assertEqual(create_spec_args["scale"], 2)  # start_scale + 1

    @patch("aind_hcr_data_transformation.compress.czi_to_zarr.write_json")
    @patch(
        "aind_hcr_data_transformation.compress."
        "czi_to_zarr.create_downsample_dataset",
        new_callable=AsyncMock,
    )
    @patch(
        "aind_hcr_data_transformation.compress." "czi_to_zarr.write_tasks",
        new_callable=AsyncMock,
    )
    @patch("aind_hcr_data_transformation.compress." "czi_to_zarr.ts.open")
    @patch(
        "aind_hcr_data_transformation.compress."
        "czi_to_zarr.czi_block_generator"
    )
    @patch("aind_hcr_data_transformation.compress." "czi_to_zarr.create_spec")
    @patch(
        "aind_hcr_data_transformation.compress."
        "czi_to_zarr._get_pyramid_metadata"
    )
    @patch(
        "aind_hcr_data_transformation.compress.czi_to_zarr"
        ".write_ome_ngff_metadata"
    )
    @patch(
        "aind_hcr_data_transformation.compress.czi_to_zarr" ".czifile.CziFile"
    )
    def test_czi_stack_zarr_writer(
        self,
        mock_czifile,
        mock_write_ome_ngff_metadata,
        mock_get_pyramid_metadata,
        mock_create_spec,
        mock_czi_block_generator,
        mock_ts_open,
        mock_write_tasks,
        mock_create_downsample_dataset,
        mock_write_json,
    ):
        """Test the czi_stack_zarr_writer function."""
        mock_logger = Mock(spec=logging.Logger)

        mock_czi = MagicMock()
        mock_czi.__enter__.return_value = mock_czi
        mock_czi.__exit__.return_value = None
        mock_czi.shape = (1, 1, 10, 20, 30)
        mock_czi.dtype = np.dtype("uint16")
        mock_czifile.return_value = mock_czi

        fake_block = np.zeros((1, 1, 10, 20, 30), dtype=np.uint16)
        mock_czi_block_generator.return_value = [(fake_block, slice(0, 10))]

        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value.write = AsyncMock()
        mock_ts_result = MagicMock()
        mock_ts_result.result.return_value = mock_dataset
        mock_ts_open.return_value = mock_ts_result

        mock_create_spec.return_value = {"mock": "spec"}
        mock_get_pyramid_metadata.return_value = {"meta": "data"}
        mock_write_ome_ngff_metadata.return_value = {"ome": "ngff"}

        # Call the function under test (it is a coroutine, so run it).
        asyncio.run(
            czi_stack_zarr_writer(
                czi_path="/fake/path/image.czi",
                output_path="/fake/output",
                voxel_size=[1.0, 1.0, 1.0],
                shard_size=[10, 10, 10],
                chunk_size=[5, 5, 5],
                scale_factor=[2, 2, 2],
                n_lvls=2,
                channel_name="DAPI",
                logger=mock_logger,
                stack_name="my_stack",
                compressor_kwargs={"cname": "zstd"},
                downsample_mode="mean",
                batch_size=4,
                bucket_name=None,
            )
        )

        # Assertions
        mock_ts_open.assert_called()
        mock_create_spec.assert_called_once()
        # write_tasks is dead code in czi_stack_zarr_writer (inlined as
        # `await dataset[region].write(...)`), so it must not be awaited.
        mock_write_tasks.assert_not_awaited()
        self.assertEqual(mock_create_downsample_dataset.await_count, 2)
        mock_write_json.assert_called_once()
        mock_logger.info.assert_called()


class TestHelpers(unittest.TestCase):
    """Test helper methods and utilities."""

    def setUp(self):
        """Set up common test data."""
        self.sample_compressor_kwargs = {
            "cname": "zstd",
            "clevel": 5,
            "shuffle": 1,
        }
        self.sample_data_shape = [1, 1, 100, 200, 300]
        self.sample_voxel_size = [2.0, 1.0, 1.0]
        self.mock_logger = Mock(spec=logging.Logger)

    def test_sample_data_consistency(self):
        """Test that sample data is consistent."""
        self.assertEqual(len(self.sample_data_shape), 5)
        self.assertEqual(len(self.sample_voxel_size), 3)
        self.assertIsInstance(self.sample_compressor_kwargs, dict)
        self.assertIn("cname", self.sample_compressor_kwargs)
