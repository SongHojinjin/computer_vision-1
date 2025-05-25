import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 데이터셋 준비 (Fashion-MNIST)
transform = transforms.ToTensor()
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)

# 공통 모델 정의 (MLP 구조)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # logits 출력
        )

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.model(x)

# 학습 함수
def train(model, loss_fn, optimizer, epochs=30, use_softmax=False, tag='', loader=None):
    model.train()
    train_losses, test_accuracies = [], []

    for epoch in range(epochs):
        total_loss = 0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)

            if use_softmax:
                output = torch.softmax(output, dim=1)
                target_onehot = torch.nn.functional.one_hot(target, num_classes=10).float()
                loss = loss_fn(output, target_onehot)
            else:
                loss = loss_fn(output, target)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        train_losses.append(total_loss / len(loader))
        acc = evaluate(model)
        test_accuracies.append(acc)

        print(f"[{tag}] Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%")

        # 시각화 (초기/중반/후반)
        if epoch in [0, 10, epochs-1]:
            print(f"[{tag}] Gradient Flow at Epoch {epoch}")
            visualize_grad_flow(model)

            print(f"[{tag}] Activation Distribution at Epoch {epoch}")
            visualize_hidden_activations(epoch, model, loader, layer_idx=1)
            visualize_hidden_activations(epoch, model, loader, layer_idx=3)

            print(f"[{tag}] Gradient Histogram at Epoch {epoch}")
            visualize_grad_flow_histogram(model)

    return train_losses, test_accuracies


# 평가 함수
def evaluate(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            preds = torch.argmax(output, dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return 100 * correct / total

#중간 레이어 활성화값 분포
def visualize_hidden_activations(epoch, model, loader, layer_idx):
    model.eval()
    activations = []
    with torch.no_grad():
        for data, _ in loader:
            x = data.view(-1, 28*28)
            for i, layer in enumerate(model.model):
                x = layer(x)
                if i == layer_idx:
                    activations.append(x.cpu().numpy())
                    break
    activations = np.concatenate(activations, axis=0)
    plt.hist(activations.flatten(), bins=100)
    plt.title(f'Layer {layer_idx} activation histogram at epoch {epoch}')
    plt.show()

# Dead ReLU 비율 히트맵
def dead_relu_ratio(tensor):
    return (tensor <= 0).sum().item() / tensor.numel()

def visualize_dead_relu(model, loader):
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            x = data.view(-1, 28*28)
            ratios = []
            for layer in model.model:
                x = layer(x)
                if isinstance(layer, nn.ReLU):
                    ratios.append(dead_relu_ratio(x))
            break
    plt.imshow([ratios], cmap="Reds", aspect='auto')
    plt.title("Dead ReLU Ratios per Layer")
    plt.colorbar(label="Ratio of inactive neurons")
    plt.yticks([])
    plt.xticks(ticks=range(len(ratios)), labels=[f"ReLU {i+1}" for i in range(len(ratios))])
    plt.show()

# Gradient 흐름 시각화 (Vanishing 여부)
def visualize_grad_flow(model):
    ave_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            ave_grads.append(param.grad.abs().mean().item())
    plt.plot(ave_grads, marker='o')
    plt.title("Average Gradient Flow per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Avg |gradient|")
    plt.grid(True)
    plt.show()

def visualize_grad_flow_histogram(model):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grads = param.grad.detach().cpu().numpy().flatten()
            plt.figure(figsize=(6, 3))
            plt.hist(grads, bins=100, alpha=0.75, color='skyblue')
            plt.title(f"Gradient Histogram: {name}")
            plt.xlabel("Gradient Value")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.show()


# CrossEntropy 모델 학습
model_ce = MLP()
loss_fn_ce = nn.CrossEntropyLoss()
optimizer_ce = optim.Adam(model_ce.parameters(), lr=0.001)
losses_ce, accs_ce = train(model_ce, loss_fn_ce, optimizer_ce, tag='CrossEntropy', loader=train_loader)

# MSE 모델 학습
model_mse = MLP()
loss_fn_mse = nn.MSELoss()
optimizer_mse = optim.Adam(model_mse.parameters(), lr=0.001)
losses_mse, accs_mse = train(model_mse, loss_fn_mse, optimizer_mse, use_softmax=True, tag='MSE', loader=train_loader)

# 그래프 시각화 (MSE)
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(losses_ce, label='CrossEntropy')
# plt.plot(losses_mse, label='MSE (with softmax)')
# plt.title("Loss vs Epoch")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
epochs = range(1, len(losses_ce)+1)

plt.figure(figsize=(14, 5))

# 🔹 Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs, losses_ce, label='CrossEntropy', color='blue')
plt.plot(epochs, losses_mse, label='MSE (with softmax)', color='orange')
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

print("\n [Dead ReLU 비율] MSE 모델")
visualize_dead_relu(model_mse, train_loader)


# 그래프 시각화 (CrossEntropy)
plt.subplot(1, 2, 2)
plt.plot(accs_ce, label='CrossEntropy')
plt.plot(accs_mse, label='MSE (with softmax)')
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.tight_layout()
plt.show()

print("\n [Dead ReLU 비율] CrossEntropy 모델")
visualize_dead_relu(model_ce, train_loader)