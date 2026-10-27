"""
Deep Learning Lab - Continuous Evaluation Program 1
---------------------------------------------------
Hugging Face Hub, dataset loading, model cards, convolution and max-pooling
in NumPy, kernel size / stride / padding, and a basic CNN on CIFAR-10.

Run locally :  python cnn_part1.py
Run on Colab:  paste the whole file into a cell (GPU runtime recommended).
"""

import os
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from huggingface_hub import dataset_info, model_info

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_TRAIN, N_VAL, N_TEST = 10_000, 2_000, 2_000
EPOCHS = 15
BATCH_SIZE = 128
LR = 1e-3


# ----------------------------------------------------------------------------
# NumPy building blocks (defined at the top so they can be reused / tested)
# ----------------------------------------------------------------------------
def output_size(n, k, s=1, p=0):
    """Output length along one dimension: floor((N + 2P - K) / S) + 1."""
    return (n + 2 * p - k) // s + 1


def conv2d_numpy(image, kernel, stride=1, padding=0):
    """Single-channel 2D convolution (cross-correlation, as in deep learning)."""
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)),
                       mode="constant", constant_values=0)
    H, W = image.shape
    kh, kw = kernel.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Kernel is larger than the (padded) input.")

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            out[i, j] = np.sum(image[r:r + kh, c:c + kw] * kernel)
    return out


def maxpool2d_numpy(image, pool_size=2, stride=None):
    """Single-channel 2D max-pooling. Stride defaults to the pool size."""
    image = np.asarray(image, dtype=np.float64)
    stride = pool_size if stride is None else stride
    H, W = image.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Pooling window is larger than the input.")

    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            out[i, j] = image[r:r + pool_size, c:c + pool_size].max()
    return out


def main():
    # ------------------------------------------------------------------------
    # PART 1: Hugging Face Hub, dataset loading and model cards
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1: Hugging Face Hub, dataset loading, model cards")
    print("=" * 70)

    # Dataset page: https://huggingface.co/datasets/uoft-cs/cifar10
    # Model page  : https://huggingface.co/microsoft/resnet-18 (example model card)
    try:
        d_info = dataset_info("uoft-cs/cifar10")
        print(f"Dataset repo : {d_info.id}")
        print(f"Downloads    : {d_info.downloads}")
        print(f"Tags         : {d_info.tags[:8]}")
    except Exception as e:  # offline or rate-limited: not fatal
        print(f"(Could not fetch dataset info from the Hub: {e})")

    try:
        m_info = model_info("microsoft/resnet-18")
        print(f"\nModel repo   : {m_info.id}")
        print(f"Pipeline tag : {m_info.pipeline_tag}")
        print(f"Library      : {m_info.library_name}")
        card = m_info.card_data
        if card is not None:
            print(f"License      : {getattr(card, 'license', None)}")
            print(f"Datasets     : {getattr(card, 'datasets', None)}")
    except Exception as e:
        print(f"(Could not fetch model info from the Hub: {e})")

    print("\nLoading CIFAR-10 from the Hub ...")
    ds = load_dataset("uoft-cs/cifar10")
    print(ds)
    class_names = ds["train"].features["label"].names
    print("Classes:", class_names)
    print("Example record keys:", list(ds["train"][0].keys()))

    # ------------------------------------------------------------------------
    # PART 2: Convolution and max-pooling in NumPy, verified on a small matrix
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: Convolution and max-pooling with NumPy")
    print("=" * 70)

    x = np.arange(25, dtype=np.float64).reshape(5, 5)
    k = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0, -1]], dtype=np.float64)  # vertical-edge kernel

    print("Input (5x5):\n", x)
    print("Kernel (3x3):\n", k)

    conv_out = conv2d_numpy(x, k, stride=1, padding=0)
    print("\nConvolution output (3x3):\n", conv_out)
    # Hand check: every column differs by 2 between c and c+2, over 3 rows
    # -> each output = 3 * (-2) = -6
    assert np.allclose(conv_out, -6), "Convolution does not match hand calculation"
    print("Hand check passed: every output equals 3 * (-2) = -6")

    t_conv = F.conv2d(torch.tensor(x).view(1, 1, 5, 5),
                      torch.tensor(k).view(1, 1, 3, 3)).squeeze().numpy()
    assert np.allclose(conv_out, t_conv), "NumPy conv differs from PyTorch"
    print("Matches torch.nn.functional.conv2d")

    p = np.array([[1, 3, 2, 4],
                  [5, 6, 1, 2],
                  [7, 2, 9, 0],
                  [3, 8, 4, 6]], dtype=np.float64)
    pool_out = maxpool2d_numpy(p, pool_size=2, stride=2)
    print("\nPooling input (4x4):\n", p)
    print("Max-pool 2x2, stride 2:\n", pool_out)
    assert np.array_equal(pool_out, np.array([[6, 4], [8, 9]])), "Max-pool is wrong"
    t_pool = F.max_pool2d(torch.tensor(p).view(1, 1, 4, 4), 2, 2).squeeze().numpy()
    assert np.allclose(pool_out, t_pool), "NumPy max-pool differs from PyTorch"
    print("Hand check passed and matches torch max_pool2d")

    # ------------------------------------------------------------------------
    # PART 3: Kernel size, stride and padding
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 3: Effect of kernel size, stride and padding")
    print("=" * 70)
    print("Formula: O = floor((N + 2P - K) / S) + 1\n")

    N = 32  # CIFAR-10 image size
    configs = [  # (kernel, stride, padding)
        (3, 1, 0), (3, 1, 1), (3, 2, 1),
        (5, 1, 0), (5, 1, 2), (5, 2, 2),
        (7, 1, 3), (7, 2, 0),
    ]
    print(f"{'N':>3} {'K':>3} {'S':>3} {'P':>3} | {'Predicted':>9} | {'Actual':>6}")
    print("-" * 40)

    # Take a sample image and convert it to grayscale for visualisation
    sample = np.array(ds["train"][0]["img"], dtype=np.float64)
    gray = sample.mean(axis=2)

    for kk, ss, pp in configs:
        predicted = output_size(N, kk, ss, pp)
        kern = np.ones((kk, kk)) / (kk * kk)  # averaging kernel
        actual = conv2d_numpy(gray, kern, stride=ss, padding=pp).shape[0]
        assert predicted == actual
        print(f"{N:>3} {kk:>3} {ss:>3} {pp:>3} | {predicted:>4}x{predicted:<4} | {actual}x{actual}")

    # Visual comparison on the sample image (edge-detection kernel)
    edge = np.array([[-1, -1, -1],
                     [-1,  8, -1],
                     [-1, -1, -1]], dtype=np.float64)
    shown = [("Original", gray, None),
             ("K=3 S=1 P=0", conv2d_numpy(gray, edge, 1, 0), None),
             ("K=3 S=1 P=1", conv2d_numpy(gray, edge, 1, 1), None),
             ("K=3 S=2 P=1", conv2d_numpy(gray, edge, 2, 1), None)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, (title, img, _) in zip(axes, shown):
        ax.imshow(img, cmap="gray")
        ax.set_title(f"{title}\n{img.shape[0]}x{img.shape[1]}")
        ax.axis("off")
    plt.suptitle("Effect of stride and padding on output size")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/conv_params_effect.png", dpi=150)
    plt.show()

    print("\nObservations:")
    print("- Larger kernel (K) -> smaller output and a wider receptive field.")
    print("- Larger stride (S) -> output shrinks roughly by a factor of S.")
    print("- Padding (P)       -> keeps borders; P=(K-1)/2 with S=1 gives 'same' size.")

    # ------------------------------------------------------------------------
    # PART 4: RGB images -> tensors, and train / validation / test subsets
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 4: RGB images to tensors; train / validation / test subsets")
    print("=" * 70)

    train_full = ds["train"].shuffle(seed=SEED)
    train_ds = train_full.select(range(N_TRAIN))                    # 10,000
    val_ds = train_full.select(range(N_TRAIN, N_TRAIN + N_VAL))     # 2,000 (disjoint)
    test_ds = ds["test"].shuffle(seed=SEED).select(range(N_TEST))   # 2,000

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 3, 1, 1)

    def to_tensors(split):
        # PIL image (32x32x3, uint8) -> float array (3x32x32) in [0, 1] -> normalised
        imgs = np.stack([np.array(im) for im in split["img"]]).astype(np.float32) / 255.0
        imgs = imgs.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        imgs = (imgs - mean) / std
        labels = np.array(split["label"], dtype=np.int64)
        return torch.from_numpy(imgs), torch.from_numpy(labels)

    X_train, y_train = to_tensors(train_ds)
    X_val, y_val = to_tensors(val_ds)
    X_test, y_test = to_tensors(test_ds)

    print(f"Train      : {tuple(X_train.shape)}  labels {tuple(y_train.shape)}")
    print(f"Validation : {tuple(X_val.shape)}  labels {tuple(y_val.shape)}")
    print(f"Test       : {tuple(X_test.shape)}  labels {tuple(y_test.shape)}")
    print("\nRoles of the subsets:")
    print("- Training   : used to update the weights.")
    print("- Validation : used to monitor generalisation / tune choices during training.")
    print("- Test       : used once at the end for an unbiased final score.")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=256)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=256)

    # ------------------------------------------------------------------------
    # PART 5: Basic CNN for CIFAR-10
    # ------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 5: Basic CNN - training and evaluation")
    print("=" * 70)

    class BasicCNN(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=3, padding=1),   # 32x32
                nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                              # 16x16
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64), nn.ReLU(),
                nn.MaxPool2d(2),                              # 8x8
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.MaxPool2d(2),                              # 4x4
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    model = BasicCNN().to(DEVICE)
    print(model)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    def run_epoch(loader, train=False):
        model.train(train)
        total_loss, correct, count = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * xb.size(0)
                correct += (logits.argmax(1) == yb).sum().item()
                count += xb.size(0)
        return total_loss / count, correct / count

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(train_loader, train=True)
        va_loss, va_acc = run_epoch(val_loader, train=False)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")

    test_loss, test_acc = run_epoch(test_loader, train=False)
    print(f"\nFinal TEST loss: {test_loss:.4f} | TEST accuracy: {test_acc:.4f}")

    # --- Learning curves -----------------------------------------------------
    epochs = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="Train")
    ax1.plot(epochs, history["val_loss"], label="Validation")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, history["train_acc"], label="Train")
    ax2.plot(epochs, history["val_acc"], label="Validation")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/learning_curves.png", dpi=150)
    plt.show()

    # --- Predictions on the test set ----------------------------------------
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            all_preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
    preds = torch.cat(all_preds).numpy()
    truth = y_test.numpy()

    # --- Confusion matrix ----------------------------------------------------
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (truth, preds), 1)  # rows = true, cols = predicted

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (test accuracy {test_acc:.2%})")
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=150)
    plt.show()

    per_class = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-class accuracy:")
    for name, acc in zip(class_names, per_class):
        print(f"  {name:<11}{acc:.3f}")

    # --- Example predictions -------------------------------------------------
    idx = np.random.RandomState(SEED).choice(len(X_test), 12, replace=False)
    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    for ax, i in zip(axes.ravel(), idx):
        img = X_test[i].numpy() * std[0] + mean[0]       # undo normalisation
        img = np.clip(img.transpose(1, 2, 0), 0, 1)
        ax.imshow(img)
        ok = preds[i] == truth[i]
        ax.set_title(f"P: {class_names[preds[i]]}\nT: {class_names[truth[i]]}",
                     color="green" if ok else "red", fontsize=9)
        ax.axis("off")
    plt.suptitle("Example predictions (green = correct, red = wrong)")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/example_predictions.png", dpi=150)
    plt.show()

    print(f"\nAll figures saved in ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
