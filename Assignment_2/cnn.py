import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Reproducibility

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Output folder

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# data normalization
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Datasets 
train_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tfms)
test_ds    = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tfms)
classes = train_full.classes

# Train/Val split
val_frac = 0.10
val_size = int(len(train_full) * val_frac)
train_size = len(train_full) - val_size
train_ds, val_ds = random_split(
    train_full, [train_size, val_size],
    generator=torch.Generator().manual_seed(seed)
)


# DataLoaders 

batch_size = 64
num_workers = 0
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Model: Custom CNN (>=3 conv layers, pooling, ReLU)
class SimpleCIFARCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # conv1
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # conv2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),  
            nn.Dropout(0.18), 

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # conv3
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),# conv4
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2), 
            nn.Dropout(0.25), 
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.32), 
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCIFARCNN().to(device)
print(model)


# Training utilities
def accuracy_from_logits(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n += 1
    return total_loss / n, total_acc / n

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n += 1
    return total_loss / n, total_acc / n

# Train config
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

epochs = 20.
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

best_val_acc = -1.0
best_state = None

print("\nStarting training...")
for epoch in range(1, epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    va_loss, va_acc = evaluate(model, val_loader, criterion)

    history["train_loss"].append(tr_loss)
    history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss)
    history["val_acc"].append(va_acc)

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"Epoch {epoch:02d}/{epochs} | "
          f"train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | "
          f"val loss {va_loss:.4f} acc {va_acc*100:.2f}%")

# Load good model
model.load_state_dict(best_state)
model.to(device)
print("\nBest val acc:", best_val_acc * 100)

ckpt_path = os.path.join(OUT_DIR, "best_model.pt")
torch.save({"model_state": model.state_dict(), "best_val_acc": best_val_acc, "history": history}, ckpt_path)
print("Saved best model to:", ckpt_path)

# Curves
epochs_axis = np.arange(1, epochs + 1)

plt.figure()
plt.plot(epochs_axis, history["train_loss"], label="train loss")
plt.plot(epochs_axis, history["val_loss"], label="val loss")
plt.xlabel("epoch"); plt.ylabel("loss"); plt.title("Loss Curves")
plt.legend()
plt.tight_layout()
loss_path = os.path.join(OUT_DIR, "loss_curves.png")
plt.savefig(loss_path, dpi=200)
plt.close()
print("Saved:", loss_path)

plt.figure()
plt.plot(epochs_axis, np.array(history["train_acc"])*100, label="train acc")
plt.plot(epochs_axis, np.array(history["val_acc"])*100, label="val acc")
plt.xlabel("epoch"); plt.ylabel("accuracy (%)"); plt.title("Accuracy Curves")
plt.legend()
plt.tight_layout()
acc_path = os.path.join(OUT_DIR, "acc_curves.png")
plt.savefig(acc_path, dpi=200)
plt.close()
print("Saved:", acc_path)

# Test accuracy
test_loss, test_acc = evaluate(model, test_loader, criterion)
print(f"\nTEST loss: {test_loss:.4f} | TEST acc: {test_acc*100:.2f}%")

# Task 2A: Feature maps from FIRST conv layer
inv_normalize = transforms.Normalize(
    mean=[-m/s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
    std=[1/s for s in CIFAR10_STD]
)

def show_image_tensor(ax, img_t, title=None):
    img = inv_normalize(img_t.cpu()).clamp(0, 1)
    ax.imshow(img.permute(1, 2, 0))
    if title:
        ax.set_title(title)
    ax.axis("off")

@torch.no_grad()
def get_first_conv_feature_maps(model, x_one):
    model.eval()
    conv1 = model.features[0]
    bn1   = model.features[1]
    relu1 = model.features[2]
    z = relu1(bn1(conv1(x_one)))
    return z

# pick 3 test images from different classes
chosen = []
seen = set()
for i in range(len(test_ds)):
    x, y = test_ds[i]
    if y not in seen:
        chosen.append((x, y, i))
        seen.add(y)
    if len(chosen) == 3:
        break

num_maps = 8

for idx, (x, y, i) in enumerate(chosen, start=1):
    x1 = x.unsqueeze(0).to(device)
    fmap = get_first_conv_feature_maps(model, x1).squeeze(0).cpu()  # (C,H,W)

    fig, axes = plt.subplots(1, num_maps + 1, figsize=(12, 3))
    show_image_tensor(axes[0], x, title=f"Input\n{classes[y]}")
    for k in range(num_maps):
        axes[k + 1].imshow(fmap[k], cmap="viridis")
        axes[k + 1].set_title(f"ch{k}")
        axes[k + 1].axis("off")

    fig.suptitle(f"Conv1 Feature Maps | test idx {i}", y=1.05)
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"task2A_featuremaps_img{idx}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

# Task 2B: Maximally activating images
# Activation = MEAN of ReLU(feature map) for filter
layer_to_probe = model.features[8] 
layer_name = "features[8] (conv3)"
activations = {}

def hook_fn(module, inp, out):
    activations["feat"] = out.detach()

hook_handle = layer_to_probe.register_forward_hook(hook_fn)

@torch.no_grad()
def score_filter_for_batch(feat, filter_idx, mode="mean"):
    feat = F.relu(feat)
    fmap = feat[:, filter_idx, :, :]
    if mode == "mean":
        return fmap.mean(dim=(1, 2))
    elif mode == "max":
        return fmap.amax(dim=(1, 2))
    else:
        raise ValueError("mode must be 'mean' or 'max'")

@torch.no_grad()
def find_topk_images_per_filter(model, loader, filter_indices, k=5, mode="mean"):
    model.eval()
    best = {fi: [] for fi in filter_indices}

    for x, y in loader:
        x = x.to(device)
        _ = model(x)
        feat = activations["feat"]

        for fi in filter_indices:
            scores = score_filter_for_batch(feat, fi, mode=mode).cpu().numpy()
            for j in range(len(scores)):
                best[fi].append((float(scores[j]), x[j].cpu(), int(y[j])))

    for fi in filter_indices:
        best[fi] = sorted(best[fi], key=lambda t: t[0], reverse=True)[:k]
    return best

selected_filters = [3, 11, 29]
activation_mode = "mean" 

topk = find_topk_images_per_filter(model, test_loader, selected_filters, k=5, mode=activation_mode)
hook_handle.remove()

for fi in selected_filters:
    items = topk[fi]
    fig, axes = plt.subplots(1, 5, figsize=(12, 3))

    for i, (score, img_t, y) in enumerate(items):
        show_image_tensor(axes[i], img_t, title=f"{classes[y]}\n{activation_mode}={score:.3f}")

    fig.suptitle(
        f"Task 2B Top-5 Activating Images | layer={layer_name} | filter={fi} | activation={activation_mode}",
        y=1.05
    )
    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"task2B_top5_filter{fi}.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

print("\nDONE. Check the outputs/ folder for plots and images.")