import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from thop import profile
from pose_dataset import PoseDataset, load_dataframes

robot_name = 'robot_3_stable_bot_l6s3'
# 'robot_1_four_ball' 'robot_2_six_ball' 'robot_3_stable_bot_l6s3' 'robot_4_robot_with_holder' 'robot_5_satble_bot_cylinder_r3l5' 'robot_6_satble_bot_cylinder_r3l6'
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
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Custom CNN builder
def build_custom_cnn(num_conv, num_filters, use_bn):
    layers = []
    in_channels = 3
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
        if use_bn:
            layers.append(nn.BatchNorm2d(num_filters))
        layers.append(nn.ReLU())
        layers.append(nn.MaxPool2d(2))
        in_channels = num_filters
    return nn.Sequential(*layers), num_filters

class SearchableCNN(nn.Module):
    def __init__(self, num_conv, num_filters, fc_dim, num_p, num_r, dropout_rate, use_bn):
        super().__init__()
        self.features, last_channels = build_custom_cnn(num_conv, num_filters, use_bn)
        self.flatten = nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            flat_size = out.view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, fc_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.head_P = nn.Linear(fc_dim, num_p)
        self.head_R = nn.Linear(fc_dim, num_r)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.head_P(x), self.head_R(x)


# Optuna objective
def objective(trial):
    num_conv = trial.suggest_int("num_conv", 1, 5)
    num_filters = trial.suggest_int("num_filters", 16, 256)
    fc_dim = trial.suggest_int("fc_dim", 64, 512)
    lr = trial.suggest_float("lr", 1e-5, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    use_bn = trial.suggest_categorical("use_bn", [True, False])
    
    model = SearchableCNN(num_conv, num_filters, fc_dim, num_p_classes, num_r_classes, dropout_rate, use_bn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=32, drop_last=True)

    # Train for 3 epochs
    for epoch in range(5):
        model.train()
        for imgs, p_labels, r_labels in train_loader:
            imgs, p_labels, r_labels = imgs.to(device), p_labels.to(device), r_labels.to(device)
            optimizer.zero_grad()
            out_P, out_R = model(imgs)
            loss = criterion(out_P, p_labels) + criterion(out_R, r_labels)
            loss.backward()
            optimizer.step()

    # Validation
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
    trial.set_user_attr("avg_acc", avg_acc)
    return 1.0 - avg_acc  # minimize loss

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Best trial
best_trial = study.best_trial
print("\n🏆 Best Configuration:")
for key, val in best_trial.params.items():
    print(f"{key}: {val}")
print(f"Best Avg Accuracy: {best_trial.user_attrs['avg_acc']:.4f}")

# Retrain best model
print("\n🔁 Retraining best model for 10 epochs...")
bp = best_trial.params
model = SearchableCNN(
    bp["num_conv"], bp["num_filters"], bp["fc_dim"],
    num_p_classes, num_r_classes,
    bp["dropout_rate"], bp["use_bn"]
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=bp["lr"])
criterion = nn.CrossEntropyLoss()
train_loader = DataLoader(train_ds, batch_size=bp["batch_size"], shuffle=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=32, drop_last=True)

for epoch in range(10):
    model.train()
    for imgs, p_labels, r_labels in train_loader:
        imgs, p_labels, r_labels = imgs.to(device), p_labels.to(device), r_labels.to(device)
        optimizer.zero_grad()
        out_P, out_R = model(imgs)
        loss = criterion(out_P, p_labels) + criterion(out_R, r_labels)
        loss.backward()
        optimizer.step()

# Test Evaluation
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

# Model stats
input_tensor = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
print(f"📦 Params: {params / 1e6:.2f}M, FLOPs: {macs / 1e9:.2f}G")

# Save model and config
torch.save(model.state_dict(), f"./checkpoint_NAS/{robot_name}/best_model.pth")
df = pd.DataFrame([bp])
df["avg_acc"] = avg_acc
df["params_m"] = round(params / 1e6, 2)
df["flops_g"] = round(macs / 1e9, 2)
df.to_csv(f"./checkpoint_NAS/{robot_name}/optuna_best_config.csv", index=False)