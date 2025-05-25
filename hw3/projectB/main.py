import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 🔹 데이터 생성 및 전처리
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.1)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_loader = torch.utils.data.DataLoader(list(zip(X_train, y_train)), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(list(zip(X_test, y_test)), batch_size=1000)

# 🔹 MLP 정의
class MLP(nn.Module):
    def __init__(self, activation='relu'):
        super(MLP, self).__init__()
        act_fn = self._get_activation(activation)
        self.activation_name = activation

        self.model = nn.Sequential(
            nn.Linear(2, 64),
            act_fn(),
            nn.Linear(64, 32),
            act_fn(),
            nn.Linear(32, 2)
        )
        self.model.apply(self._init_weights)

    def _get_activation(self, name):
        if name == 'relu':
            return nn.ReLU
        elif name == 'leakyrelu':
            return lambda: nn.LeakyReLU(0.01)
        elif name == 'sigmoid':
            return nn.Sigmoid
        else:
            raise ValueError("Unsupported activation function")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.model(x)

# 🔹 평가 함수
def evaluate(model):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    return 100. * correct / len(test_loader.dataset)

# 🔹 학습 함수
def train(model, loss_fn, optimizer, epochs=30, tag='', loader=None):
    train_losses, test_accuracies = [], []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for data, target in loader:
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss / len(loader))
        acc = evaluate(model)
        test_accuracies.append(acc)

        print(f"[{tag}] Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={acc:.2f}%")
        
        if epoch in [0, 10, epochs-1]:
            visualize_hidden_activations(epoch, model, loader, 1)
            visualize_hidden_activations(epoch, model, loader, 3)
            print(f"[{tag}] Dead Neuron Heatmap at Epoch {epoch}")
            visualize_dead_neurons(model, loader, model.activation_name, tag=f"{tag}-Epoch{epoch}")
    
    return train_losses, test_accuracies

# 활성화 함수에 따른 dead neuron 비율 계산 함수
def dead_neuron_ratio(tensor, act_fn_name):
    if act_fn_name == 'relu':
        return (tensor <= 0).sum().item() / tensor.numel()
    elif act_fn_name == 'leakyrelu':
        # LeakyReLU는 음수도 통과되지만 거의 0에 가까운 값이 많음 → 동일 기준 사용
        return (tensor <= 0).sum().item() / tensor.numel()
    elif act_fn_name == 'sigmoid':
        # Sigmoid는 saturation zone (0.01 이하 또는 0.99 이상) 기준
        return ((tensor < 0.01) | (tensor > 0.99)).sum().item() / tensor.numel()
    else:
        return 0.0

# 히트맵으로 시각화 (모든 활성화 함수용)
def visualize_dead_neurons(model, loader, act_fn_name, tag=''):
    model.eval()
    with torch.no_grad():
        for data, _ in loader:
            x = data.view(x.size(0), -1)
            ratios = []
            for layer in model.model:
                x = layer(x)
                if isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.Sigmoid)):
                    ratio = dead_neuron_ratio(x, act_fn_name)
                    ratios.append(ratio)
            break  # 한 배치만 사용

    if ratios:
        plt.figure(figsize=(6, 2))
        plt.imshow([ratios], cmap="Reds", aspect='auto')
        plt.title(f"{tag} Dead Neuron Ratio (per Activation Layer)")
        plt.colorbar(label="Dead Neuron Ratio")
        plt.xticks(ticks=range(len(ratios)), labels=[f"Layer {i+1}" for i in range(len(ratios))])
        plt.yticks([])
        plt.tight_layout()
        plt.show()

# 🔹 중간 활성화 히스토그램
def visualize_hidden_activations(epoch, model, loader, layer_idx):
    model.eval()
    activations = []
    with torch.no_grad():
        for data, _ in loader:
            x = data
            for i, layer in enumerate(model.model):
                x = layer(x)
                if i == layer_idx:
                    activations.append(x.cpu().numpy())
                    break
    activations = np.concatenate(activations, axis=0)
    plt.hist(activations.flatten(), bins=100)
    plt.title(f'Layer {layer_idx} activation histogram at epoch {epoch}')
    plt.show()

# 🔁 실험 반복 실행 (ReLU, LeakyReLU, Sigmoid)
results = {}
for act in ['relu', 'leakyrelu', 'sigmoid']:
    print(f"\n========== Activation: {act.upper()} ==========")
    model = MLP(activation=act)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    losses, accs = train(model, loss_fn, optimizer, tag=act.upper(), loader=train_loader)
    results[act] = {'loss': losses, 'acc': accs}

# 🔎 성능 비교 그래프
epochs = range(1, len(next(iter(results.values()))['loss'])+1)
plt.figure(figsize=(14, 5))

# 🔹 Loss
plt.subplot(1, 2, 1)
for act in results:
    plt.plot(epochs, results[act]['loss'], label=act.upper())
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# 🔹 Accuracy
plt.subplot(1, 2, 2)
for act in results:
    plt.plot(epochs, results[act]['acc'], label=act.upper())
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
