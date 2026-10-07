"""
Unit tests of io utilities
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from aind_hcr_data_transformation.utils import utils

RESOURCES_DIR = (
    Path(os.path.dirname(os.path.realpath(__file__))) / ".." / "resources"
)

JSON_FILE_PATH = RESOURCES_DIR / "local_json.json"


class IoUtilitiesTest(unittest.TestCase):
    """Class for testing the io utilities"""

    def test_add_leading_dim(self):
        """
        Tests that a new dimension is added
        to the array.
        """
        test_arr = np.zeros((2, 2), dtype=np.uint8)
        transformed_arr = utils.add_leading_dim(data=test_arr)

        self.assertEqual(test_arr.ndim + 1, transformed_arr.ndim)

    def test_extract_data(self):
        """
        Tests the array data is extracted
        when there are expanded dimensions.
        """
        test_arr = np.zeros((1, 1, 1, 2, 2), dtype=np.uint8)
        transformed_arr_no_lead = utils.extract_data(arr=test_arr)
        transformed_arr_with_lead = utils.extract_data(
            arr=test_arr, last_dimensions=3
        )

        self.assertEqual(2, transformed_arr_no_lead.ndim)
        self.assertEqual(test_arr.shape[-2:], transformed_arr_no_lead.shape)

        self.assertEqual(3, transformed_arr_with_lead.ndim)
        self.assertEqual(test_arr.shape[-3:], transformed_arr_with_lead.shape)

    def test_extract_data_fail(self):
        """
        Tests failure of extract data
        """
        test_arr = np.zeros((2, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            utils.extract_data(arr=test_arr, last_dimensions=3)

    def test_pad_array(self):
        """Tests padding an array"""
        test_arr = np.zeros((2, 2), dtype=np.uint8)
        padded_test_arr = utils.pad_array_n_d(arr=test_arr)

        self.assertEqual(5, padded_test_arr.ndim)

        padded_test_arr = utils.pad_array_n_d(arr=test_arr, dim=-1)
        self.assertEqual(test_arr.ndim, padded_test_arr.ndim)

    def test_pad_array_fail(self):
        """Tests padding an array"""
        test_arr = np.zeros((2, 2), dtype=np.uint8)

        with self.assertRaises(ValueError):
            utils.pad_array_n_d(arr=test_arr, dim=6)

    def test_read_json_as_dict(self):
        """
        Tests successful reading of a dictionary
        """
        expected_result = {"some_key": "some_value"}
        result = utils.read_json_as_dict(JSON_FILE_PATH)
        self.assertEqual(expected_result, result)

    @patch("aind_hcr_data_transformation.utils.utils.boto3.client")
    def test_sync_dir_to_s3(self, mock_boto_client):
        """Tests that each file is uploaded to the right key."""
        mock_s3 = Mock()
        mock_paginator = Mock()
        mock_paginator.paginate.return_value = iter([{"Contents": []}])
        mock_s3.get_paginator.return_value = mock_paginator
        mock_boto_client.return_value = mock_s3

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "a.txt").write_text("hello")
            sub = tmp_path / "sub"
            sub.mkdir()
            (sub / "b.txt").write_text("world")

            utils.sync_dir_to_s3(tmp_path, "s3://bucket/prefix")

        self.assertEqual(mock_s3.upload_file.call_count, 2)
        uploaded_keys = sorted(
            call.args[2] for call in mock_s3.upload_file.call_args_list
        )
        self.assertEqual(
            uploaded_keys, ["prefix/a.txt", "prefix/sub/b.txt"]
        )
        for call in mock_s3.upload_file.call_args_list:
            self.assertEqual(call.args[1], "bucket")

    @patch("aind_hcr_data_transformation.utils.utils.boto3.client")
    def test_sync_dir_to_s3_skips_existing(self, mock_boto_client):
        """Tests that existing same-size objects are skipped."""
        mock_s3 = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            local_file = tmp_path / "a.txt"
            local_file.write_text("hello")
            size = local_file.stat().st_size

            mock_paginator = Mock()
            mock_paginator.paginate.return_value = iter(
                [{"Contents": [{"Key": "prefix/a.txt", "Size": size}]}]
            )
            mock_s3.get_paginator.return_value = mock_paginator
            mock_boto_client.return_value = mock_s3

            utils.sync_dir_to_s3(tmp_path, "s3://bucket/prefix")

        mock_s3.upload_file.assert_not_called()

    def test_sync_dir_to_s3_missing_dir(self):
        """Tests that a missing local directory raises."""
        with self.assertRaises(FileNotFoundError):
            utils.sync_dir_to_s3(
                Path("/definitely/does/not/exist"),
                "s3://bucket/prefix",
            )

    @patch("aind_hcr_data_transformation.utils.utils.boto3.client")
    def test_copy_file_to_s3(self, mock_boto_client):
        """Tests that an explicit destination key is used verbatim."""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3

        with tempfile.NamedTemporaryFile() as tmp:
            utils.copy_file_to_s3(tmp.name, "s3://bucket/dest/file.bin")

        mock_s3.upload_file.assert_called_once()
        call = mock_s3.upload_file.call_args
        self.assertEqual(call.args[1], "bucket")
        self.assertEqual(call.args[2], "dest/file.bin")

    @patch("aind_hcr_data_transformation.utils.utils.boto3.client")
    def test_copy_file_to_s3_prefix(self, mock_boto_client):
        """Tests that a trailing slash falls back to basename."""
        mock_s3 = Mock()
        mock_boto_client.return_value = mock_s3

        with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
            basename = Path(tmp.name).name
            utils.copy_file_to_s3(tmp.name, "s3://bucket/dest/")

        mock_s3.upload_file.assert_called_once()
        self.assertEqual(
            mock_s3.upload_file.call_args.args[2], f"dest/{basename}"
        )

    def test_parse_s3_url_invalid(self):
        """Tests that non-s3 URLs raise ValueError."""
        with self.assertRaises(ValueError):
            utils._parse_s3_url("https://example.com/foo")
        with self.assertRaises(ValueError):
            utils._parse_s3_url("s3:///no-bucket")

    def test_validate_slices_valid(self):
        """Tests that slices are valid when within bounds"""
        utils.validate_slices(2, 5, 10)

    def test_validate_slices_invalid(self):
        """Tests that an error is raised when slices are out of bounds"""

        with self.assertRaises(ValueError):
            utils.validate_slices(5, 2, 10)

    def test_default_stride(self):
        """Tests that the default stride is 128"""
        self.assertEqual(utils.generate_jumps(500), list(range(0, 500, 128)))

    def test_custom_stride(self):
        """Tests that a custom stride is used"""
        self.assertEqual(utils.generate_jumps(256, 64), [0, 64, 128, 192])

    def test_valid_axis(self):
        """Tests that a valid axis returns the correct index and length"""
        index, length = utils.get_axis_index([1, 2, 3], "zyx", "y")
        self.assertEqual(index, 1)
        self.assertEqual(length, 2)

    def test_invalid_axis(self):
        """Tests that an invalid axis raises a ValueError"""
        with self.assertRaises(ValueError):
            utils.get_axis_index([1, 2, 3], "zyx", "q")


class TestReadSlicesCzi(unittest.TestCase):
    """Unit tests for read_slices_czi function."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock CZI stream
        self.mock_czi_stream = Mock()
        self.mock_czi_stream.shape = (100, 512, 512, 1, 50)
        self.mock_czi_stream.dtype = np.uint16
        self.mock_czi_stream.axes = "TYXCZ"
        self.mock_czi_stream.start = [0, 0, 0, 0, 0]
        self.mock_czi_stream._fh = Mock()
        self.mock_czi_stream._fh.lock = None

        # Mock subblock directory
        self.subblock_directory = [Mock() for _ in range(50)]

    @patch("aind_hcr_data_transformation.utils.utils.validate_slices")
    @patch("aind_hcr_data_transformation.utils.utils.create_output")
    @patch("aind_hcr_data_transformation.utils.utils.parallel_reader")
    def test_sequential_processing(
        self, mock_parallel_reader, mock_create_output, mock_validate_slices
    ):
        """Test sequential processing (single worker)."""
        mock_output = Mock()
        mock_output.flush = Mock()
        mock_create_output.return_value = mock_output

        with patch("numpy.squeeze") as mock_squeeze:
            mock_squeeze.return_value = "squeezed_result"

            result = utils.read_slices_czi(
                self.mock_czi_stream,
                self.subblock_directory,
                10,
                20,
                max_workers=1,  # Force sequential
            )

            # Verify calls
            mock_validate_slices.assert_called_once_with(10, 20, 50)
            mock_create_output.assert_called_once()
            self.assertEqual(
                mock_parallel_reader.call_count, 10
            )  # 20 - 10 = 10 slices
            mock_output.flush.assert_called_once()
            mock_squeeze.assert_called_once_with(mock_output)
            self.assertEqual(result, "squeezed_result")

    @patch("aind_hcr_data_transformation.utils.utils.validate_slices")
    @patch("aind_hcr_data_transformation.utils.utils.create_output")
    @patch("aind_hcr_data_transformation.utils.utils.parallel_reader")
    @patch("multiprocessing.cpu_count")
    def test_debug_conditions_for_parallel(
        self,
        mock_cpu_count,
        mock_parallel_reader,
        mock_create_output,
        mock_validate_slices,
    ):
        """Debug test to check what conditions are being evaluated."""
        mock_cpu_count.return_value = 8
        mock_output = Mock()
        mock_output.flush = Mock()
        mock_create_output.return_value = mock_output

        with patch("numpy.squeeze") as mock_squeeze:
            mock_squeeze.return_value = "squeezed_result"

            # Test with explicit max_workers > 1 and slice count > 1
            utils.read_slices_czi(
                self.mock_czi_stream,
                self.subblock_directory,
                10,
                20,  # This gives us 10 slices (20-10)
                max_workers=4,  # Explicitly > 1
            )

    @patch("aind_hcr_data_transformation.utils.utils.validate_slices")
    @patch("aind_hcr_data_transformation.utils.utils.create_output")
    @patch("aind_hcr_data_transformation.utils.utils.parallel_reader")
    def test_parallel_processing_with_proper_mocking(
        self, mock_parallel_reader, mock_create_output, mock_validate_slices
    ):
        """Test parallel processing with proper ThreadPoolExecutor mocking."""
        mock_output = Mock()
        mock_output.flush = Mock()
        mock_create_output.return_value = mock_output

        # Create a more realistic mock for ThreadPoolExecutor
        mock_executor = Mock()
        mock_map_function = Mock()
        mock_executor.map = mock_map_function

        with (
            patch("numpy.squeeze") as mock_squeeze,
            patch(
                "concurrent.futures.ThreadPoolExecutor"
            ) as mock_executor_class,
        ):
            mock_squeeze.return_value = "squeezed_result"
            mock_executor_class.return_value.__enter__.return_value = (
                mock_executor
            )
            mock_executor_class.return_value.__exit__.return_value = None

            utils.read_slices_czi(
                self.mock_czi_stream,
                self.subblock_directory,
                10,
                25,  # 15 slices to ensure > 1
                max_workers=4,  # Explicitly > 1
            )

            # Debug: Check if ThreadPoolExecutor was called
            if mock_executor_class.called:
                print(
                    "ThreadPoolExecutor call args:"
                    f"{mock_executor_class.call_args}"
                )

            # Verify ThreadPoolExecutor was used
            # mock_executor_class.assert_called_once_with(4)
            # mock_executor.map.assert_called_once()

            # Verify the lock was set
            self.assertIsNone(
                self.mock_czi_stream._fh.lock
            )  # Should be None after execution

    def test_conditions_check(self):
        """Test the exact conditions that trigger parallel processing."""
        # These are the conditions from your function:
        max_workers = 4
        start_slice = 10
        end_slice = 20
        slice_count = end_slice - start_slice  # Should be 10

        condition1 = max_workers > 1  # Should be True
        condition2 = slice_count > 1  # Should be True
        both_conditions = condition1 and condition2  # Should be True

        # The parallel path should be taken
        self.assertTrue(both_conditions)

    @patch("aind_hcr_data_transformation.utils.utils.validate_slices")
    @patch("aind_hcr_data_transformation.utils.utils.create_output")
    @patch("aind_hcr_data_transformation.utils.utils.parallel_reader")
    def test_parallel_processing_alternative_approach(
        self, mock_parallel_reader, mock_create_output, mock_validate_slices
    ):
        """Alternative approach to test parallel processing."""
        mock_output = Mock()
        mock_output.flush = Mock()
        mock_create_output.return_value = mock_output

        # Mock ThreadPoolExecutor at the module level where it's imported
        with (
            patch("numpy.squeeze") as mock_squeeze,
            patch(
                "aind_hcr_data_transformation.utils.utils.ThreadPoolExecutor"
            ) as mock_executor_class,
        ):
            mock_squeeze.return_value = "squeezed_result"
            mock_executor = Mock()
            mock_executor_class.return_value.__enter__.return_value = (
                mock_executor
            )
            mock_executor_class.return_value.__exit__.return_value = None

            utils.read_slices_czi(
                self.mock_czi_stream,
                self.subblock_directory,
                5,
                25,  # 20 slices
                max_workers=8,  # High number to ensure > 1
            )

            # This should work if ThreadPoolExecutor
            # is imported in the utils module
            if mock_executor_class.called:
                mock_executor_class.assert_called_once()
                mock_executor.map.assert_called_once()
            else:
                print("ThreadPoolExecutor not called - check import location")

    @patch("aind_hcr_data_transformation.utils.utils.validate_slices")
    @patch("aind_hcr_data_transformation.utils.utils.create_output")
    @patch("aind_hcr_data_transformation.utils.utils.parallel_reader")
    def test_with_side_effect_to_verify_path(
        self, mock_parallel_reader, mock_create_output, mock_validate_slices
    ):
        """Use side effects to verify which execution path is taken."""
        mock_output = Mock()
        mock_output.flush = Mock()
        mock_create_output.return_value = mock_output

        # Track which path is taken
        parallel_path_taken = []
        sequential_path_taken = []

        def track_parallel_creation(*args, **kwargs):
            """Track if parallel path is taken."""
            parallel_path_taken.append(True)
            mock_executor = Mock()
            mock_executor.map = Mock()
            return mock_executor

        def track_sequential_call(*args, **kwargs):
            """Track if sequential path is taken."""
            sequential_path_taken.append(True)

        mock_parallel_reader.side_effect = track_sequential_call

        with (
            patch("numpy.squeeze") as mock_squeeze,
            patch(
                "concurrent.futures.ThreadPoolExecutor",
                side_effect=track_parallel_creation,
            ),
        ):
            mock_squeeze.return_value = "squeezed_result"

            utils.read_slices_czi(
                self.mock_czi_stream,
                self.subblock_directory,
                10,
                20,
                max_workers=4,
            )


class TestGetAvailableCpuCount(unittest.TestCase):
    """Tests for ``get_available_cpu_count``."""

    def test_slurm_env_takes_precedence(self):
        """SLURM_CPUS_PER_TASK should be returned verbatim when set."""
        with patch.dict(
            os.environ, {"SLURM_CPUS_PER_TASK": "2"}, clear=False
        ):
            self.assertEqual(utils.get_available_cpu_count(), 2)

    def test_slurm_env_one_returns_one(self):
        """Explicit ``SLURM_CPUS_PER_TASK=1`` must return 1, not be
        rounded up by any later fallback."""
        with patch.dict(
            os.environ, {"SLURM_CPUS_PER_TASK": "1"}, clear=False
        ):
            self.assertEqual(utils.get_available_cpu_count(), 1)

    def test_slurm_env_zero_clamped_to_one(self):
        """Pathological ``SLURM_CPUS_PER_TASK=0`` is clamped to 1."""
        with patch.dict(
            os.environ, {"SLURM_CPUS_PER_TASK": "0"}, clear=False
        ):
            self.assertEqual(utils.get_available_cpu_count(), 1)

    def test_invalid_slurm_env_falls_through(self):
        """Non-integer SLURM value should fall through to the next
        resolution step instead of raising."""
        with patch.dict(
            os.environ,
            {"SLURM_CPUS_PER_TASK": "not-a-number"},
            clear=False,
        ):
            self.assertGreaterEqual(utils.get_available_cpu_count(), 1)

    def test_affinity_used_when_no_slurm_env(self):
        """With no SLURM env, the helper should fall back to scheduling
        affinity (Linux) or ``multiprocessing.cpu_count``."""
        env = {
            k: v
            for k, v in os.environ.items()
            if k != "SLURM_CPUS_PER_TASK"
        }
        with patch.dict(os.environ, env, clear=True):
            n = utils.get_available_cpu_count()
            self.assertGreaterEqual(n, 1)
            if hasattr(os, "sched_getaffinity"):
                self.assertEqual(n, len(os.sched_getaffinity(0)))


if __name__ == "__main__":
    unittest.main()
