"""Tests for the SmartSPIM data transfer"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aind_hcr_data_transformation.models import (
    ZeissJobSettings,
)
from aind_hcr_data_transformation.zeiss_job import (
    ZeissCompressionJob,
)


class ZarrCompressionTest(unittest.TestCase):
    """Class for testing the data transform"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""
        cls.raw_data_folder = tempfile.mkdtemp(prefix="unittest_")
        cls.temp_folder = tempfile.mkdtemp(prefix="unittest_")

        basic_job_settings = ZeissJobSettings(
            input_source=Path(cls.raw_data_folder),
            output_directory=Path(cls.temp_folder),
            num_of_partitions=4,
            partition_to_process=0,
        )
        cls.basic_job_settings = basic_job_settings
        cls.basic_job = ZeissCompressionJob(job_settings=basic_job_settings)

    @patch.object(ZeissCompressionJob, "run_job", return_value=None)
    def test_run_job(self, mock_run_job):
        """Tests Zeiss compression and zarr writing"""
        self.basic_job.run_job()
        mock_run_job.assert_called_once()

    @classmethod
    def tearDownClass(cls) -> None:
        """Tear down class method to clean up"""
        if os.path.exists(cls.temp_folder):
            shutil.rmtree(cls.temp_folder, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
