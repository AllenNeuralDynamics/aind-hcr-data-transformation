"""Tests for the SmartSPIM data transfer"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from natsort import natsorted
from numcodecs.blosc import Blosc

from aind_hcr_data_transformation.models import ZeissJobSettings
from aind_hcr_data_transformation.zeiss_job import ZeissCompressionJob

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"


class ZeissCompressionTest(unittest.TestCase):
    """Class for testing the data transform"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""
        cls.raw_data_folder = tempfile.mkdtemp(prefix="unittest_")

        cls.n_dummy_czis = 11
        # Create dummy .czi files
        for i in range(cls.n_dummy_czis):  # X goes from 0 to 10
            file_path = Path(cls.raw_data_folder) / f"488_large({i}).czi"
            file_path.touch()  # Creates an empty file

        basic_job_settings = ZeissJobSettings(
            input_source=Path(cls.raw_data_folder),
            output_directory="fake_output_dir",
            num_of_partitions=4,
            partition_to_process=0,
        )
        cls.basic_job_settings = basic_job_settings
        cls.basic_job = ZeissCompressionJob(job_settings=basic_job_settings)

    def test_partition_list(self):
        """Tests partition list method"""
        test_list = [f"ID: {x}" for x in range(75)]
        output_list1 = self.basic_job.partition_list(
            test_list, num_of_partitions=5
        )
        output_list2 = self.basic_job.partition_list(
            test_list, num_of_partitions=2
        )
        flat_output1 = [x for xs in output_list1 for x in xs]
        flat_output2 = [x for xs in output_list2 for x in xs]
        self.assertEqual(5, len(output_list1))
        self.assertEqual(2, len(output_list2))
        self.assertCountEqual(test_list, flat_output1)
        self.assertCountEqual(test_list, flat_output2)

    def test_get_partitioned_list_of_stack_paths(self):
        """Tests _get_partitioned_list_of_stack_paths"""
        stack_paths = self.basic_job._get_partitioned_list_of_stack_paths()
        flat_list_of_paths = natsorted(
            [x.stem for xs in stack_paths for x in xs]
        )
        expected_flat_list = [
            f"488_large({i})" for i in range(self.n_dummy_czis)
        ]
        self.assertEqual(4, len(stack_paths))
        self.assertEqual(expected_flat_list, flat_list_of_paths)

    def test_get_compressor(self):
        """Tests _get_compressor method"""

        compressor = self.basic_job._get_compressor()
        expected_compressor = Blosc(
            cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE, blocksize=0
        )
        self.assertEqual(expected_compressor, compressor)

    def test_get_compressor_none(self):
        """Tests _get_compressor method returns None if no config set"""

        job_settings = ZeissJobSettings.model_construct(
            input_source="", output_directory="", compressor_name="foo"
        )
        job = ZeissCompressionJob(job_settings=job_settings)
        compressor = job._get_compressor()
        self.assertIsNone(compressor)

    @patch.object(ZeissCompressionJob, "run_job", return_value=None)
    def test_run_job(self, mock_run_job):
        """Tests Zeiss compression and zarr writing"""
        self.basic_job.run_job()
        mock_run_job.assert_called_once()


if __name__ == "__main__":
    unittest.main()
