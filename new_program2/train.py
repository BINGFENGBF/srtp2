from matplotlib.pylab import f
import torch

def train_one_epoch(model,loader,criterion,optimizer,device,log_interval=10):
    model.train()

    total_loss=0.0
    total_samples=0
    correct=0

    for batch_idx,(imgs,labels) in enumerate(loader,start=1):#?
        imgs=imgs.to(device)
        labels=labels.to(device)

        outputs=model(imgs)
        loss=criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()*imgs.size(0)
        total_samples+=imgs.size(0) 

        preds=outputs.argmax(dim=1)
        correct+=(preds==labels).sum().item()

        if batch_idx % log_interval==0:
            print(f"  Batch {batch_idx}: loss {loss.item():.4f}, acc {(preds==labels).float().mean().item():.4f}")

    return total_loss/total_samples, correct/total_samples#平均损失和准确率

