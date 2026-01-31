import os
from google.cloud import storage


def download_from_bucket(bucket_name, source_blob_name, destination_file_name):
    """从 GCS 下载文件 | Downloads a file from GCS."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # 确保本地 data/ 目录存在 | Ensure local data/ directory exists
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)

    blob.download_to_filename(destination_file_name)
    print(f"✅ Success: {source_blob_name} downloaded to {destination_file_name}")


if __name__ == "__main__":
    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    # 将云端的 raw_data/sample.txt 下载为 data/downloaded_sample.txt
    download_from_bucket(BUCKET_NAME, "raw_data/sample.txt", "data/downloaded_sample.txt")