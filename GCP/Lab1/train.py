import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


def main():
    # 1. 参数解析：Vertex AI 会自动传入这些路径
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default=os.environ.get('AIP_STORAGE_URI'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('AIP_MODEL_DIR'))
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()

    # 2. 设备检测：确保 GPU ⚡ 正常工作
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. 数据加载 (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 4. 极简模型定义 (用于验证流程)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()  # 清空梯度 🧹
            output = model(data)
            loss = criterion(output, target)
            loss.backward()  # 反向传播 ⚙️
            optimizer.step()  # 更新参数 🦶
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # 6. 保存模型到 GCS 🪣
    # 生产实践：仅保存 state_dict
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    save_path = os.path.join(args.model_dir, "mnist_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()