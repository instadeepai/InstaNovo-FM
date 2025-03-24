import os
import shutil
from s3fs.core import S3FileSystem
from pathlib import Path
from dreams.utils.data import MSData


def download_data(data_path, bucket_location):
    """Download data file from S3 to local filesystem."""
    if "S3_ENDPOINT" not in os.environ:
        print("S3_ENDPOINT not set. Returning data_path.")
        return data_path

    # Initialize S3 filesystem
    s3 = S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")})

    buckets = {
        "input": os.environ["AICHOR_INPUT_PATH"],   
        "output": 's3://',
    }
    # Define source and destination paths
    source_path = os.path.join(
        buckets[bucket_location], data_path
    )  # don't use Pathlib for s3 paths

    destination_dir = Path(
        "/home/appuser/mass_spectrometry_foundation_model/dreams/data"
    )
    destination_path = destination_dir / Path(data_path).name

    # Ensure destination directory exists
    destination_dir.mkdir(parents=True, exist_ok=True)

    # Read and save the file from S3
    print(f"Downloading {source_path}")
    with s3.open(source_path, "rb") as s3_file, destination_path.open(
        mode="wb"
    ) as local_file:
        shutil.copyfileobj(s3_file, local_file)

    print(f"File downloaded to {destination_path}")
    return destination_path


def convert(mgf_path):
    """Converts mgf file to hdf5 format."""
    print(f"Saving MSData to HDF5 for {mgf_path}")
    hdf5_path = Path(mgf_path).with_suffix(".hdf5")

    # converts to hdf5
    MSData.from_mgf(mgf_path, in_mem=False)
    assert hdf5_path.exists(), f"Something went wrong. {hdf5_path} not found"
    print(f"Saved MSData to {hdf5_path}")
    return hdf5_path


def upload_hdf5(hdf5_path, subfolder):
    """Uploads hdf5 file to S3."""
    if "S3_ENDPOINT" not in os.environ:
        return hdf5_path

    s3 = S3FileSystem(client_kwargs={"endpoint_url": os.environ.get("S3_ENDPOINT")})

    destination_dir = os.path.join(os.environ["AICHOR_OUTPUT_PATH"], subfolder)

    # Ensure destination directory exists
    os.makedirs(destination_dir, exist_ok=True)

    destination_path = os.path.join(destination_dir, hdf5_path.name)  # s3 path

    # Read and save the file from S3
    with open(hdf5_path, "rb") as local_file, s3.open(
        destination_path, "wb"
    ) as s3_file:
        shutil.copyfileobj(local_file, s3_file)

    print(f"File uploaded to {destination_path}")
    return destination_path


if __name__ == "__main__":
    # mgf_path = "denovo_dataset_v1_mgf/train.mgf"
    # destination_path = download_mgf(mgf_path)
    # hdf5_path = convert(destination_path)
    # upload_hdf5(hdf5_path, "denovo_dataset_v1_hdf5")
    # print("Done")
    download_data("mass-spectro-194096dec8b74901-outputs/output/7f5f9429-e729-481f-9bbb-d5039406e015/denovo_dataset_v1_hdf5/train.hdf5", "output")
