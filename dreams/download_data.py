import os
import shutil
from s3fs.core import S3FileSystem


def download_hdf5():
    """Download sample file."""
    # Initialize S3 filesystem
    s3 = S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")})

    # Define source and destination paths
    source_path = os.path.join(
        os.environ["AICHOR_INPUT_PATH"], "DreaMS/20210408-Paleofeces-2604-01.hdf5"
    )
    destination_dir = "dreams/data"
    destination_path = os.path.join(destination_dir, "20210408-Paleofeces-2604-01.hdf5")

    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    # Read and save the file from S3
    with s3.open(source_path, "rb") as s3_file, open(
        destination_path, "wb"
    ) as local_file:
        shutil.copyfileobj(s3_file, local_file)

    print(f"File downloaded to {destination_path}")


if __name__ == "__main__":
    download_hdf5()
