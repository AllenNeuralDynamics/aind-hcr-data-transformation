"""
Example job with partitioning
"""

import sys

from aind_protein_data_transformation.models import ZeissJobSettings
from aind_protein_data_transformation.zeiss_job import ZeissCompressionJob


def main(partition_id):
    """
    Example job with partition ID
    """
    num_partitions = 1  # Should match the number of nodes

    basic_job_settings = ZeissJobSettings(
        input_source="/path/to/data/tiles_test",
        output_directory=f"./test_conversion_partition_{partition_id}",
        num_of_partitions=num_partitions,
        partition_to_process=partition_id,
        target_size_mb=19200,
        s3_location="bucket",
    )
    basic_job = ZeissCompressionJob(job_settings=basic_job_settings)
    basic_job.run_job()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script.py <partition_id>")
        sys.exit(1)

    partition_id = int(sys.argv[1])
    main(partition_id)
