import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision import models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from thop import profile
import pandas as pd
from pose_model_zoo import MultiHeadWrapper, SimpleCNN, build_backbone_model
from pose_dataset import PoseDataset, load_dataframes

# Settings
robot_name = 'robot_1_four_ball'
# 'robot_1_four_ball' 'robot_2_six_ball' 'robot_3_stable_bot_l6s3' 'robot_4_robot_with_holder' 'robot_5_satble_bot_cylinder_r3l5' 'robot_6_satble_bot_cylinder_r3l6'
model_names = ["cnn", "vgg16", "resnet18", "resnet50", "efficientnet", "mobilenetv2", "vit"]
# data_sizes = ["min", 500, 400, 300, 200, 100, 50, 25, 10]
data_sizes = [16]
os.makedirs(f"./checkpoint/{robot_name}", exist_ok=True)
data_root_folder = f"./Image_Pose/{robot_name}"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training loop
def train_model(model, loader, val_loader, optimizer, criterion, model_path_prefix, epochs=10):
    best_score = -1.0
    best_model = None
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, p_labels, r_labels in loader:
            imgs, p_labels, r_labels = imgs.to(device), p_labels.to(device), r_labels.to(device)
            optimizer.zero_grad()
            out_P, out_R = model(imgs)
            loss_P = criterion(out_P, p_labels)
            loss_R = criterion(out_R, r_labels)
            loss = loss_P + loss_R
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} - Total Loss: {total_loss:.4f}")
        # Save latest checkpoint
        torch.save(model.state_dict(), f"{model_path_prefix}_latest.pth")

        # Evaluate on val
        p_metrics, r_metrics = evaluate(model, val_loader)
        avg_acc = (p_metrics[0] + r_metrics[0]) / 2
        print(f"          Average Acc: {avg_acc}")
        if avg_acc >= best_score:
            best_score = avg_acc
            best_model = model.state_dict()
            torch.save(best_model, f"{model_path_prefix}_best.pth")

    if best_model:
        model.load_state_dict(best_model)
    return model


# Evaluation
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_true_P, all_pred_P, all_true_R, all_pred_R = [], [], [], []
    for imgs, p_labels, r_labels in loader:
        imgs = imgs.to(device)
        out_P, out_R = model(imgs)
        all_true_P.extend(p_labels.numpy())
        all_true_R.extend(r_labels.numpy())
        all_pred_P.extend(out_P.argmax(1).cpu().numpy())
        all_pred_R.extend(out_R.argmax(1).cpu().numpy())

    def get_metrics(true, pred):
        return [
            accuracy_score(true, pred),
            precision_score(true, pred, average='macro', zero_division=0),
            recall_score(true, pred, average='macro', zero_division=0),
            f1_score(true, pred, average='macro', zero_division=0),
        ]
    return get_metrics(all_true_P, all_pred_P), get_metrics(all_true_R, all_pred_R)


# Main experiment
results = []
train_df, val_df, test_df, num_p_classes, num_r_classes = load_dataframes(root_path=data_root_folder)

for size in data_sizes:
    min_count = train_df.groupby(['P', 'R']).size().min()
    limit = min_count if size == "min" else size

    sampled_df = train_df.groupby(['P', 'R']).apply(lambda x: x.sample(n=min(len(x), limit), random_state=42)).reset_index(drop=True)
    sampled_ds = PoseDataset(sampled_df)
    val_ds = PoseDataset(val_df)
    test_ds = PoseDataset(test_df)
    print(f"Train: {len(sampled_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = DataLoader(sampled_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)
    test_loader = DataLoader(test_ds, batch_size=16)

    for model_name in model_names:
        print(f"\n🚀 Training {model_name.upper()} with {limit} imgs/class")
        if model_name == "cnn":
            model = SimpleCNN(num_p_classes, num_r_classes).to(device)
        else:
            model = build_backbone_model(model_name, num_p_classes, num_r_classes, pretrained=True).to(device)

        lr = 1e-4 if model_name in ["vgg16", "vit"] else 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        model_path_prefix = f"./checkpoint/{robot_name}/{model_name}_{limit}"
        

        model = train_model(model, train_loader, val_loader, optimizer, criterion, model_path_prefix, epochs=10)
        p_metrics, r_metrics = evaluate(model, test_loader)

        # Params + FLOPs using thop
        input_tensor = torch.randn(1, 3, 224, 224).to(device)
        macs, params = profile(model, inputs=(input_tensor,), verbose=False)

        results.append({
            "Model": model_name,
            "Data_per_class": limit,
            "P_Acc": p_metrics[0], "P_Prec": p_metrics[1], "P_Rec": p_metrics[2], "P_F1": p_metrics[3],
            "R_Acc": r_metrics[0], "R_Prec": r_metrics[1], "R_Rec": r_metrics[2], "R_F1": r_metrics[3],
            "Params_M": round(params / 1e6, 2),
            "FLOPs_G": round(macs / 1e9, 2)
        })
        print({"Model": model_name,
            "Data_per_class": limit,
            "P_Acc": p_metrics[0], "P_Prec": p_metrics[1], "P_Rec": p_metrics[2], "P_F1": p_metrics[3],
            "R_Acc": r_metrics[0], "R_Prec": r_metrics[1], "R_Rec": r_metrics[2], "R_F1": r_metrics[3],
            "Params_M": round(params / 1e6, 2),
            "FLOPs_G": round(macs / 1e9, 2)})
        print("✅ Done.")

# Save results
output_csv = f"./checkpoint/{robot_name}/results_{robot_name}_{limit}.csv"
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"\n📊 All results saved to {output_csv}")
