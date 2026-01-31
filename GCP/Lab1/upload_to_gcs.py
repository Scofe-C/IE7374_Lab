import os
from google.cloud import storage


def upload_to_bucket(bucket_name, source_file_name, destination_blob_name):
    """ Uploads a file to the GCS bucket."""
    # 初始化客户端，它会自动使用 GitHub Actions 提供的认证信息
    storage_client = storage.Client()

    # 检查存储桶是否存在，不存在则创建
    try:
        bucket = storage_client.get_bucket(bucket_name)
    except Exception:
        print(f"Bucket {bucket_name} not found, creating it...")
        bucket = storage_client.create_bucket(bucket_name)

    # 上传文件
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

    print(f"Success: {source_file_name} uploaded to {bucket_name}/{destination_blob_name}")


if __name__ == "__main__":

    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "my-ml-data-bucket")

    upload_to_bucket(BUCKET_NAME, "data/sample.txt", "raw_data/sample.txt")