from collections import Counter, defaultdict
import csv
import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision


def load_items_from_csv(csv_path:str,name_key="id",label_key="label"):
    items=[]
    with open(csv_path,"r",encoding="utf-8")as f:
        reader=csv.DictReader(f)
        for row in reader:
            fname=row[name_key]
            y=int(row[label_key])
            items.append((fname,y))
    return items

def stratified_split(items,val_ratio=0.2,seed=18):
    random.seed(seed)
    buckets=defaultdict(list)
    for fname,y in items:
        buckets[y].append(fname)

    train_items=[],val_items=[]
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
    print(name,"total=",total,"dist=",{k:(v,v/total) for k,v in c.items()})


class CsvListDataset(Dataset):
    def __init__(self,root_dir,items,transforms=None,suffix=""):
        self.root_dir=root_dir
        self.items=items
        self.transforms=transforms
        self.suffix=suffix

    def __len__(self):
        return len(self.items)
    

    def __getitem__(self,idx):   #针对单个样本的读取
        fname,y=self.items[idx]
        img_path=os.path.join(self.root_dir,fname+self.suffix)
        img=Image.open(img_path).convert("RGB")
        if self.transforms:
            img=self.transforms(img)
        return img,y
    
def build_transforms(train:bool):
    if train:
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(90),
            torchvision.transforms.ToTensor(),
            torchvision.tranforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])#from ImageNet(效果可能不太好)
        ])
    
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.tranforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])#from ImageNet
    ])

