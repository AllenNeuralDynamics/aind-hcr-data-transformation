"""Tests for the Z1 data transfer"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from natsort import natsorted

from aind_hcr_data_transformation.models import ZeissJobSettings
from aind_hcr_data_transformation.zeiss_job import ZeissCompressionJob

RESOURCES_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "resources"


class ZeissCompressionTest(unittest.TestCase):
    """Class for testing the data transform"""

    @classmethod
    def setUpClass(cls) -> None:
        """Setup basic job settings and job that can be used across tests"""
        cls.raw_data_folder = Path(tempfile.mkdtemp(prefix="unittest_"))
        cls.raw_data_folder = cls.raw_data_folder.joinpath("SPIM")
        cls.raw_data_folder.mkdir()

        cls.n_dummy_czis = 11
        # Create dummy .czi files
        for i in range(cls.n_dummy_czis):  # X goes from 0 to 10
            file_path = Path(cls.raw_data_folder) / f"488_large({i}).czi"
            file_path.touch()  # Creates an empty file

        basic_job_settings = ZeissJobSettings(
            input_source=Path(cls.raw_data_folder).parent,
            output_directory="fake_output_dir",
            num_of_partitions=4,
            partition_to_process=0,
        )
        cls.basic_job_settings = basic_job_settings
        cls.basic_job = ZeissCompressionJob(job_settings=basic_job_settings)

    @patch("aind_hcr_data_transformation.utils.utils.read_json_as_dict")
    def test_valid_acquisition_file(self, mock_read_json):
        """
        Tests that the voxel resolution is correctly
        extracted from a valid acquisition.json file
        """
        mock_read_json.return_value = {
            "schema_version": "1.0.0",
            "tiles": [
                {
                    "coordinate_transformations": [
                        {"type": "translation", "translation": [1, 2, 3]},
                        {"type": "scale", "scale": [0.5, 0.4, 0.3]},
                    ]
                }
            ],
        }

        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = True

        result = ZeissCompressionJob._get_voxel_resolution(mock_path)
        self.assertEqual(result, [0.3, 0.4, 0.5])

    @patch("aind_hcr_data_transformation.utils.utils.read_json_as_dict")
    def test_valid_acquisition2_file(self, mock_read_json):
        """
        Tests that the voxel resolution is correctly
        extracted from a valid acquisition.json file
        """
        mock_read_json.return_value = {
            "schema_version": "2.0.0",
            "data_streams": [
                {
                    "configurations": [
                        {
                            "images": [
                                {
                                    "image_to_acquisition_transform": [
                                        {
                                            "object_type": "Scale",
                                            "scale": [0.5, 0.4, 0.3],
                                        },
                                        {
                                            "object_type": "Translation",
                                            "translation": [1, 2, 3],
                                        },
                                    ]
                                },
                                {
                                    "image_to_acquisition_transform": [
                                        {
                                            "object_type": "Scale",
                                            "scale": [0.5, 0.4, 0.3],
                                        },
                                        {
                                            "object_type": "Translation",
                                            "translation": [1, 2, 3],
                                        },
                                    ]
                                },
                            ]
                        }
                    ]
                },
            ],
        }

        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = True

        result = ZeissCompressionJob._get_voxel_resolution(mock_path)
        self.assertEqual(result, [0.3, 0.4, 0.5])

    # test real acquisition file in tests/resources/acquisition_2.0.json
    def test_real_acquisition_2_file(self):
        """
        Tests that the voxel resolution is correctly extracted
        from a real acquisition_2.0.json file in the resources directory.
        """
        acquisition_path = RESOURCES_DIR / "acquisition_2.0.json"
        self.assertTrue(
            acquisition_path.is_file(),
            "acquisition_2.0.json does not exist in resources directory",
        )
        result = ZeissCompressionJob._get_voxel_resolution(acquisition_path)
        # Update the expected value below to match the actual expected voxel
        #  size in your test file
        expected_voxel_size = [1, 0.22936919442229586, 0.22936919442229586]
        self.assertEqual(result, expected_voxel_size)

    def test_missing_file(self):
        """
        Tests that a FileNotFoundError is raised
        if the acquisition file is missing
        """
        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = False

        with self.assertRaises(FileNotFoundError):
            ZeissCompressionJob._get_voxel_resolution(mock_path)

    @patch("aind_hcr_data_transformation.utils.utils.read_json_as_dict")
    def test_missing_scale(self, mock_read_json):
        """Tests that an IndexError is raised if no scale is present"""
        mock_read_json.return_value = {
            "schema_version": "1.0.0",
            "tiles": [
                {
                    "coordinate_transformations": [
                        {"type": "translation", "translation": [1, 2, 3]}
                    ]
                }
            ],
        }

        mock_path = MagicMock(spec=Path)
        mock_path.is_file.return_value = True

        with self.assertRaises(IndexError):  # [0] access fails if no scale
            ZeissCompressionJob._get_voxel_resolution(mock_path)

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
        expected_compressor = {
            "cname": "zstd",
            "clevel": 3,
            "shuffle": "shuffle",
        }
        self.assertEqual(expected_compressor, compressor)

    def test_get_compressor_none(self):
        """Tests _get_compressor method returns None if no config set"""

        job_settings = ZeissJobSettings.model_construct(
            input_source="", output_directory="", compressor_name="foo"
        )
        job = ZeissCompressionJob(job_settings=job_settings)
        compressor = job._get_compressor()
        self.assertIsNone(compressor)

    @patch("aind_hcr_data_transformation.utils.utils.sync_dir_to_s3")
    @patch("pathlib.Path")
    def test_no_s3_location(self, mock_path_cls, mock_sync):
        """
        Tests _upload_derivatives_folder
        when no S3 location is set
        """
        instance = MagicMock()
        instance.job_settings = MagicMock()
        instance.job_settings.s3_location = None
        instance.job_settings.input_source = "/local/data"

        mock_derivatives_path = MagicMock()
        mock_derivatives_path.exists.return_value = True
        mock_path_cls().joinpath.return_value = mock_derivatives_path

        instance._upload_derivatives_folder()
        mock_sync.assert_not_called()

    @patch.object(ZeissCompressionJob, "run_job", return_value=None)
    def test_run_job(self, mock_run_job):
        """Tests Zeiss compression and zarr writing"""
        self.basic_job.run_job()
        mock_run_job.assert_called_once()

    def test_czi_reader_max_workers_default_is_none(self):
        """``czi_reader_max_workers`` defaults to ``None`` so existing
        callers retain the auto-sized thread-pool behavior."""
        self.assertIsNone(
            self.basic_job_settings.czi_reader_max_workers
        )

    def test_czi_reader_max_workers_can_be_set(self):
        """A user can force serial CZI reads (``max_workers=1``) to work
        around the known ``czifile`` thread-safety segfault."""
        settings = ZeissJobSettings(
            input_source=Path(self.raw_data_folder).parent,
            output_directory="fake_output_dir",
            num_of_partitions=4,
            partition_to_process=0,
            czi_reader_max_workers=1,
        )
        self.assertEqual(settings.czi_reader_max_workers, 1)

    @patch(
        "aind_hcr_data_transformation.zeiss_job.czi_stack_zarr_writer",
        new_callable=AsyncMock,
    )
    @patch.object(
        ZeissCompressionJob, "_get_voxel_resolution", return_value=[1, 1, 1]
    )
    def test_write_stacks_forwards_czi_reader_max_workers(
        self, _mock_voxel, mock_writer
    ):
        """``_write_stacks`` must forward ``czi_reader_max_workers`` to the
        underlying writer so the setting actually reaches the CZI reader."""
        settings = ZeissJobSettings(
            input_source=Path(self.raw_data_folder).parent,
            output_directory="fake_output_dir",
            s3_location="s3://bucket/prefix",
            num_of_partitions=1,
            partition_to_process=0,
            czi_reader_max_workers=1,
        )
        job = ZeissCompressionJob(job_settings=settings)
        job._write_stacks([Path("/fake/tile.czi")])
        mock_writer.assert_called_once()
        self.assertEqual(
            mock_writer.call_args.kwargs["czi_reader_max_workers"], 1
        )


if __name__ == "__main__":
    unittest.main()
