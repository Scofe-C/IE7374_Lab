import os
from google.cloud import aiplatform

# 1. 初始化 SDK | Initialize the SDK
# 这里连接到你的 GCP 项目和区域
aiplatform.init(
    project='my-learning-001-486001',
    location='us-central1',
    staging_bucket='gs://ie7374-lab-data-my-learning-001-486001'
)

# 2. 定义自定义作业 | Define the Custom Job
# 我们告诉 Vertex AI 使用哪种机器、哪个镜像以及运行哪个脚本
job = aiplatform.CustomJob.from_local_script(
    display_name="mnist_pytorch_gpu_job",
    script_path="GCP/Lab1/train.py", # 确保路径正确
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest",
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    # 传递给 train.py 的参数 | Arguments passed to train.py
    args=[
        "--data-dir", "gs://ie7374-lab-data-my-learning-001-486001/mnist",
        "--model-dir", "gs://ie7374-lab-data-my-learning-001-486001/models",
        "--epochs", "5"
    ],
)

# 3. 提交并运行 | Submit and Run
# sync=True 会让你的终端保持开启直到任务完成，方便查看实时日志
print(" Submitting Custom Job to Vertex AI...")
job.run(sync=True)
print(" Job completed!")
