import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms
from thop import profile
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pose_dataset import PoseDataset, load_dataframes

robot_name = 'robot_3_stable_bot_l6s3'
# 'robot_1_four_ball' 'robot_2_six_ball' 'robot_3_stable_bot_l6s3' 'robot_4_robot_with_holder' 'robot_5_satble_bot_cylinder_r3l5' 'robot_6_satble_bot_cylinder_r3l6'
config_path = f"./checkpoint_NAS/robot_1_four_ball/optuna_best_config.csv"
os.makedirs(f"./checkpoint_NAS_Pose/{robot_name}", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
size = 16

# Load best config
cfg = pd.read_csv(config_path).iloc[0].to_dict()

# Dataset
train_df, val_df, test_df, num_p_classes, num_r_classes = load_dataframes(root_path=f"./Image_Pose/{robot_name}")
min_count = train_df.groupby(['P', 'R']).size().min()
limit = min_count if size == "min" else size

sampled_df = train_df.groupby(['P', 'R']).apply(lambda x: x.sample(n=min(len(x), limit), random_state=42)).reset_index(drop=True)
train_ds = PoseDataset(sampled_df)
val_ds = PoseDataset(val_df)
test_ds = PoseDataset(test_df)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=32, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=32, drop_last=True)

# Model
class SearchableCNN(nn.Module):
    def __init__(self, num_conv, num_filters, fc_dim, num_p, num_r, dropout_rate, use_bn):
        super().__init__()
        layers = []
        in_channels = 3
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1))
            if use_bn:
                layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = num_filters
        self.features = nn.Sequential(*layers)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = self.features(dummy)
            flat_size = out.view(1, -1).shape[1]

        self.flatten = nn.Flatten()
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

model = SearchableCNN(
    num_conv=int(cfg["num_conv"]),
    num_filters=int(cfg["num_filters"]),
    fc_dim=int(cfg["fc_dim"]),
    num_p=num_p_classes,
    num_r=num_r_classes,
    dropout_rate=float(cfg["dropout_rate"]),
    use_bn=bool(cfg["use_bn"])
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))
criterion = nn.CrossEntropyLoss()

# Train
def train_model(model, loader, val_loader, optimizer, criterion, epochs=10):
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for imgs, p_labels, r_labels in loader:
            imgs, p_labels, r_labels = imgs.to(device), p_labels.to(device), r_labels.to(device)
            optimizer.zero_grad()
            out_P, out_R = model(imgs)
            loss = criterion(out_P, p_labels) + criterion(out_R, r_labels)
            loss.backward()
            optimizer.step()

        acc_P, acc_R = evaluate(model, val_loader)[0][0], evaluate(model, val_loader)[1][0]
        avg_acc = (acc_P + acc_R) / 2
        print(f"Epoch {epoch+1}, Val Avg Acc: {avg_acc:.4f}")
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save(model.state_dict(), f"./checkpoint_NAS_Pose/{robot_name}/best_model_retrained.pth")

# Evaluate
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_true_P, all_pred_P, all_true_R, all_pred_R = [], [], [], []
    for imgs, p_labels, r_labels in loader:
        imgs = imgs.to(device)
        out_P, out_R = model(imgs)
        all_true_P.extend(p_labels.numpy())
        all_pred_P.extend(out_P.argmax(1).cpu().numpy())
        all_true_R.extend(r_labels.numpy())
        all_pred_R.extend(out_R.argmax(1).cpu().numpy())

    def get_metrics(true, pred):
        return [
            accuracy_score(true, pred),
            precision_score(true, pred, average='macro', zero_division=0),
            recall_score(true, pred, average='macro', zero_division=0),
            f1_score(true, pred, average='macro', zero_division=0)
        ]

    return get_metrics(all_true_P, all_pred_P), get_metrics(all_true_R, all_pred_R)

# Run
train_model(model, train_loader, val_loader, optimizer, criterion)
model.load_state_dict(torch.load(f"./checkpoint_NAS_Pose/{robot_name}/best_model_retrained.pth"))
p_metrics, r_metrics = evaluate(model, test_loader)
print("\n✅ Test Results:")
print(f"P_Acc: {p_metrics[0]:.4f}, R_Acc: {r_metrics[0]:.4f}, Avg: {(p_metrics[0]+r_metrics[0])/2:.4f}")

# Params + FLOPs using thop
input_tensor = torch.randn(1, 3, 224, 224).to(device)
macs, params = profile(model, inputs=(input_tensor,), verbose=False)
results = []
results.append({
    "P_Acc": p_metrics[0], "P_Prec": p_metrics[1], "P_Rec": p_metrics[2], "P_F1": p_metrics[3],
    "R_Acc": r_metrics[0], "R_Prec": r_metrics[1], "R_Rec": r_metrics[2], "R_F1": r_metrics[3],
    "Params_M": round(params / 1e6, 2),
    "FLOPs_G": round(macs / 1e9, 2)
})
print({
    "P_Acc": p_metrics[0], "P_Prec": p_metrics[1], "P_Rec": p_metrics[2], "P_F1": p_metrics[3],
    "R_Acc": r_metrics[0], "R_Prec": r_metrics[1], "R_Rec": r_metrics[2], "R_F1": r_metrics[3],
    "Params_M": round(params / 1e6, 2),
    "FLOPs_G": round(macs / 1e9, 2)})
print("✅ Done.")

# Save results
output_csv = f"./checkpoint_NAS_Pose/{robot_name}/results_{robot_name}.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n📊 All results saved to {output_csv}")