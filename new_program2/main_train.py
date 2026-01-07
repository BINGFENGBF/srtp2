import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

from datasets import load_items_from_csv, stratified_split, show_dist, CsvListDataset, build_transforms
from model2 import build_model, freeze_backbone
from train import train_one_epoch
from eval import eval_one_epoch


def main():
    # 路径
    csv_path = "Dataset/train_labels.csv"
    img_root = "Dataset/train"
    suffix = ".tif"

    # 超参
    batch_size = 64
    num_epochs = 5
    lr = 1e-3
    pretrained = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 数据
    items = load_items_from_csv(csv_path)
    train_items, val_items = stratified_split(items, val_ratio=0.2, seed=42)
    show_dist("TRAIN", train_items)
    show_dist("VAL", val_items)

    train_ds = CsvListDataset(img_root, train_items, transforms=build_transforms(train=True), suffix=suffix)
    val_ds = CsvListDataset(img_root, val_items, transforms=build_transforms(train=False), suffix=suffix)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 模型
    model = build_model(num_classes=2, pretrained=pretrained, dropout_p=0.5)
    freeze_backbone(model)  # 迁移学习：先只训头
    model.to(device)

    # loss：最简单的“类别不平衡补丁”
    # 注：这里权重只是示例。你可以用你的真实比例做更合理的数。
    class_weight = torch.tensor([1.0, 4.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    # 只优化可训练参数（被冻结的不会更新）
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir="runs/exp_v2")

    best_val_acc = 0.0
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_one_epoch(model, val_loader, criterion, device)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        print(f"Epoch {epoch}: train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f}")

        # 保存最好模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best.pth")

    writer.close()
    print("best val acc:", best_val_acc)


if __name__ == "__main__":
    main()
