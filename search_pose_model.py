import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args
from torch.utils.data import DataLoader
from pose_dataset import PoseDataset, load_dataframes
from sklearn.metrics import accuracy_score
from thop import profile

robot_name = 'robot_1_four_ball'
data_root_folder = f"./Image_Pose/{robot_name}"
os.makedirs(f"./checkpoint_NAS/{robot_name}", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = 16

# Load data
train_df, val_df, test_df, num_p_classes, num_r_classes = load_dataframes(root_path=data_root_folder)
min_count = train_df.groupby(['P', 'R']).size().min()
limit = min_count if size == "min" else size

sampled_df = train_df.groupby(['P', 'R']).apply(lambda x: x.sample(n=min(len(x), limit), random_state=42)).reset_index(drop=True)
train_ds = PoseDataset(sampled_df)
val_ds = PoseDataset(val_df)
test_ds = PoseDataset(test_df)
print(f"Train: {len(sampled_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Custom CNN builder
def build_custom_cnn(num_conv, num_filters, fc_dim):
    layers = []
    in_channels = 3
    for _ in range(num_conv):
        layers += [
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]
        in_channels = num_filters

    return nn.Sequential(*layers), num_filters

class SearchableCNN(nn.Module):
    def __init__(self, num_conv, num_filters, fc_dim, num_p, num_r):
        super().__init__()
        self.features, last_channels = build_custom_cnn(num_conv, num_filters, fc_dim)
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(last_channels * 28 * 28, fc_dim),
            nn.ReLU()
        )
        self.head_P = nn.Linear(fc_dim, num_p)
        self.head_R = nn.Linear(fc_dim, num_r)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.head_P(x), self.head_R(x)

# Search space
space  = [
    Integer(1, 5, name='num_conv'),
    Integer(16, 256, name='num_filters'),
    Integer(64, 512, name='fc_dim'),
    Real(1e-5, 1e-2, prior='log-uniform', name='lr'),
    Integer(8, 32, name='batch_size')
]

best_cfg = {}
best_score = -1

@use_named_args(space)
def objective(**params):
    global best_cfg, best_score
    print(f"\n🔍 Trying: {params}")
    model = SearchableCNN(
        num_conv=params['num_conv'],
        num_filters=params['num_filters'],
        fc_dim=params['fc_dim'],
        num_p=num_p_classes,
        num_r=num_r_classes
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    # Train for multiple epochs
    model.train()
    for epoch in range(3):
        for imgs, p_labels, r_labels in train_loader:
            imgs, p_labels, r_labels = imgs.to(device), p_labels.to(device), r_labels.to(device)
            optimizer.zero_grad()
            out_P, out_R = model(imgs)
            loss = criterion(out_P, p_labels) + criterion(out_R, r_labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_true_P, all_pred_P, all_true_R, all_pred_R = [], [], [], []
    with torch.no_grad():
        for imgs, p_labels, r_labels in val_loader:
            imgs = imgs.to(device)
            out_P, out_R = model(imgs)
            all_true_P.extend(p_labels.numpy())
            all_pred_P.extend(out_P.argmax(1).cpu().numpy())
            all_true_R.extend(r_labels.numpy())
            all_pred_R.extend(out_R.argmax(1).cpu().numpy())

    acc_P = accuracy_score(all_true_P, all_pred_P)
    acc_R = accuracy_score(all_true_R, all_pred_R)
    avg_acc = (acc_P + acc_R) / 2
    print(f"✅ Avg Accuracy: {avg_acc:.4f} (P: {acc_P:.4f}, R: {acc_R:.4f})")

    if avg_acc > best_score:
        best_score = avg_acc
        best_cfg = params.copy()

    return -avg_acc

# Run Bayesian Optimization
res = gp_minimize(
    func=objective,
    dimensions=space,
    n_calls=20,
    random_state=42,
    verbose=True
)

# Final train best model and report FLOPs/Params
print("\n🏆 Best configuration:")
for name, val in best_cfg.items():
    print(f"{name}: {val}")

model = SearchableCNN(
    num_conv=best_cfg['num_conv'],
    num_filters=best_cfg['num_filters'],
    fc_dim=best_cfg['fc_dim'],
    num_p=num_p_classes,
    num_r=num_r_classes
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_cfg['lr'])
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_ds, batch_size=best_cfg['batch_size'], shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
test_loader = DataLoader(test_ds, batch_size=32)

# Retrain
print("\n🔁 Retraining best model...")
for epoch in range(10):
    model.train()
    for imgs, p_labels, r_labels in train_loader:
        imgs, p_labels, r_labels = imgs.to(device), p_labels.to(device), r_labels.to(device)
        optimizer.zero_grad()
        out_P, out_R = model(imgs)
        loss = criterion(out_P, p_labels) + criterion(out_R, r_labels)
        loss.backward()
        optimizer.step()

# Evaluate on test
model.eval()
all_true_P, all_pred_P, all_true_R, all_pred_R = [], [], [], []
with torch.no_grad():
    for imgs, p_labels, r_labels in test_loader:
        imgs = imgs.to(device)
        out_P, out_R = model(imgs)
        all_true_P.extend(p_labels.numpy())
        all_pred_P.extend(out_P.argmax(1).cpu().numpy())
        all_true_R.extend(r_labels.numpy())
        all_pred_R.extend(out_R.argmax(1).cpu().numpy())

acc_P = accuracy_score(all_true_P, all_pred_P)
acc_R = accuracy_score(all_true_R, all_pred_R)
avg_acc = (acc_P + acc_R) / 2
print(f"\n🧪 Test Accuracy - P: {acc_P:.4f}, R: {acc_R:.4f}, Avg: {avg_acc:.4f}")

# Report FLOPs and Params
input_tensor = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
print(f"📦 Params: {params / 1e6:.2f}M, FLOPs: {macs / 1e9:.2f}G")

# Save best config
pd.DataFrame([best_cfg]).to_csv(f"./checkpoint_NAS/{robot_name}/best_config.csv", index=False)