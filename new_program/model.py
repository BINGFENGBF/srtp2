import csv
import random
from collections import defaultdict
from collections import Counter
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


#1.提取items
csv_path="Dataset/train_labels.csv"#√
name_key="id"
label_key="label"#这两行删了都没关系

items=[] #初始化一个列表,储存(fname, y)元组
with open("Dataset/train_labels.csv","r",encoding="utf-8")as f:
    reader=csv.DictReader(f)

    # print("CSV列名:",reader.fieldnames)

    for row in reader:
        fname=row["id"]
        y=int(row["label"])
        items.append((fname,y))#row,fname,y都可以随便取名,id,label是csv里的列名,必须用csv里的列的名字


# print("总样本数:",len(items))
# print("前5条items:",items[:5])

# count0=sum(1 for _,y in items if y==0)
# count1=sum(1 for _,y in items if y==1)
# print("标签0的样本数:",count0)
# print("标签1的样本数:",count1)


#划分数据集
def stratified_split(items,val_ratio=0.2,seed=42):
    random.seed(seed)

    buckets=defaultdict(list)

    for fname,y in items:
        buckets[y].append(fname)

    train_items=[]
    val_items=[]

    for y,fnames in buckets.items():
        random.shuffle(fnames)

        n_val=int(len(fnames)*val_ratio)

        val_f=fnames[:n_val]

        train_f=fnames[n_val:]

        val_items+=[(f,y) for f in val_f]
        train_items+=[(f,y) for f in train_f]

    random.shuffle(train_items)
    random.shuffle(val_items)

    return train_items,val_items


def show_dist(name,items):
    c=Counter([y for _,y in items])
    total=len(items)

    print(
        name,
        "total=",
        total,
        "dist=",
        {k:(v,v/total) for k,v in c.items()}
    )

train_items,val_items=stratified_split(items,val_ratio=0.2,seed=42)

show_dist("ALL", items)
show_dist("TRAIN", train_items)
show_dist("VAL", val_items)

print("TRAIN前3条:",train_items[:3])
print("VAL前3条:",val_items[:3])



#dataset和dataloader

suffix=".tif"

transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

class CsvListDataset(Dataset):
    def __init__(self,root_dir,items,transforms=None,suffix=""):
        self.root_dir=root_dir
        self.items=items
        self.transforms=transforms
        self.suffix=suffix

    def __len__(self):
        return len(self.items)

    def __getitem__(self,idx):
        fname,y=self.items[idx]
        img_path=os.path.join(self.root_dir,fname+self.suffix)

        img=Image.open(img_path).convert("RGB")
        if self.transforms:
            img=self.transforms(img)

        return img,y


train_ds=CsvListDataset("Dataset/train",train_items,transforms=transform,suffix=suffix)
val_ds=CsvListDataset("Dataset/train",val_items,transforms=transform,suffix=suffix)

train_loader=DataLoader(train_ds,batch_size=64,shuffle=True,num_workers=0)
val_loader=DataLoader(val_ds,batch_size=64,shuffle=False,num_workers=0)

x,y=next(iter(train_loader))
print("一个batch的x形状:",x.shape,"y形状:",y.shape)
print("y前10个:",y[:10])

#设备
device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
print("当前设备:",device)

#加载Resnet34,暂时没加载预训练(现在加)
model=models.resnet34(weights=None)
model=models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

#修改最后一层为2类
model.fc=nn.Linear(model.fc.in_features,2)

#把模型放在device上
model.to(device)

#定义loss
loss=nn.CrossEntropyLoss()

#定义优化器
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)


#开始训练
def train_one_epoch(model,loader,criterion,optimizer,device):
    total_loss=0.0
    total_samples=0
    tp=0
    fp=0
    fn=0
    tn=0

    for batch_idx,(imgs,labels) in enumerate(loader, start=1):
        #device
        imgs=imgs.to(device)
        labels=labels.to(device)

        #前向传播
        outputs=model(imgs)
        loss=criterion(outputs,labels)

        #反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #统计
        total_loss+=loss.item()*imgs.size(0)
        preds=outputs.argmax(dim=1)
        total_samples+=imgs.size(0)

        tp+=((preds==1)&(labels==1)).sum().item()
        fp+=((preds==1)&(labels==0)).sum().item()
        fn+=((preds==0)&(labels==1)).sum().item()
        tn+=((preds==0)&(labels==0)).sum().item()

        #指标
        if batch_idx % 10 ==0:
            batch_loss=loss.item()
            batch_acc=(preds==labels).float().mean().item()
            print(f"  Batch {batch_idx}: loss {batch_loss:.4f}, acc {batch_acc:.4f}")

    avg_loss=total_loss/total_samples
    total=tp+fp+fn+tn
    acc=(tp+tn)/total if total>0 else 0.0
    precision=tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return avg_loss,acc,precision,recall,f1

#验证
def eval_one_epoch(model,loader,criterion,device):
    model.eval()

    total_loss=0.0
    total_samples=0
    tp=0
    fp=0
    fn=0
    tn=0

    with torch.no_grad():
        for imgs,labels in loader:
            imgs=imgs.to(device)
            labels=labels.to(device)

            outputs=model(imgs)
            loss=criterion(outputs,labels)

            total_loss+=loss.item()*imgs.size(0)
            preds=outputs.argmax(dim=1)
            total_samples+=imgs.size(0)

            tp+=((preds==1)&(labels==1)).sum().item()
            fp+=((preds==1)&(labels==0)).sum().item()
            fn+=((preds==0)&(labels==1)).sum().item()
            tn+=((preds==0)&(labels==0)).sum().item()

    avg_loss=total_loss/total_samples
    total=tp+fp+fn+tn
    acc=(tp+tn)/total if total>0 else 0.0
    precision=tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall=tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1=2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
    return avg_loss,acc,precision,recall,f1

#真正开始训练

num_epochs=5

writer=SummaryWriter(log_dir="runs/exp1")

for epoch in range(num_epochs):
    train_loss,train_acc,train_p,train_r,train_f1=train_one_epoch(model,train_loader,loss,optimizer,device)
    val_loss,val_acc,val_p,val_r,val_f1=eval_one_epoch(model,val_loader,loss,device)

    writer.add_scalar("Loss/train", train_loss, epoch+1)
    writer.add_scalar("Loss/val", val_loss, epoch+1)
    writer.add_scalar("Acc/train", train_acc, epoch+1)
    writer.add_scalar("Acc/val", val_acc, epoch+1)
    writer.add_scalar("Precision/train", train_p, epoch+1)
    writer.add_scalar("Precision/val", val_p, epoch+1)
    writer.add_scalar("Recall/train", train_r, epoch+1)
    writer.add_scalar("Recall/val", val_r, epoch+1)
    writer.add_scalar("F1/train", train_f1, epoch+1)
    writer.add_scalar("F1/val", val_f1, epoch+1)

    print(
        f"Epoch[{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} "
        f"P: {train_p:.4f} R: {train_r:.4f} F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} "
        f"P: {val_p:.4f} R: {val_r:.4f} F1: {val_f1:.4f}"
    )

    torch.save(model.state_dict(),f"model_epoch{epoch+1}.pth")

writer.close()
