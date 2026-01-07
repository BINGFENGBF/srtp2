import torch


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

    return total_loss / total_samples, correct / total_samples
