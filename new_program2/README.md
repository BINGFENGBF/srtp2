# ResNet34+PCAM加强版

## 项目介绍

这是一个基于 ResNet34 的医学图像二分类项目，使用迁移学习和带权重的交叉熵损失函数来处理类别不平衡问题。项目针对 PCAM 数据集进行优化，用于组织学图像的自动分类。

## 项目结构

new_program2/
├── main_train.py      # 主训练脚本
├── train.py           # 训练循环实现
├── eval.py            # 评估循环实现
├── model.py           # 模型定义与构建
├── datasets.py        # 数据集处理与加载
└── README.md          # 项目文档

## 主要文件说明

### main_train.py

- **功能**：项目主入口，配置所有超参数和运行流程
- **关键功能**：
  - 数据加载与分层划分（8:2 的训练/验证集）
  - 模型初始化和冻结主干网络（迁移学习）
  - 设置加权交叉熵损失处理类别不平衡

### model.py

- **build_model()**：构建 ResNet34 模型
  - 支持预训练权重加载
  - Dropout 层防止过拟合
  - 自定义全连接层用于二分类
- **freeze_backbone()**：冻结除全连接层外的所有参数
- **unfreeze_backbone()**：解冻所有参数

### datasets.py

- **load_items_from_csv()**：从 CSV 文件加载图像-标签对
- **stratified_split()**：按类别比例分层划分数据集
- **show_dist()**：显示数据集的类别分布
- **CsvListDataset**：自定义 PyTorch Dataset 类

### train.py

- **train_one_epoch()**：单个 epoch 的训练过程
  - 前向传播、损失计算、反向传播
  - 实时输出损失和准确率

### eval.py

- **eval_one_epoch()**：单个 epoch 的验证过程
  - 无梯度计算（评估模式）
  - 返回平均损失和准确率

## 主要特性

- ✅ **迁移学习**：使用 ImageNet 预训练的 ResNet34
- ✅ **动态冻结**：先冻结主干网络，只训练分类头
- ✅ **类别平衡**：加权交叉熵损失处理类别不平衡（权重: 1.0, 4.0）
- ✅ **分层划分**：按类别比例分割训练/验证集，避免采样偏差
- ✅ **CUDA 支持**：自动检测 GPU 并使用

## 依赖环境

torch>=1.9.0
torchvision>=0.10.0
pillow>=8.0.0
tensorboard>=2.5.0

## 使用方法

### 1. 准备数据集

确保数据组织如下：

Dataset/
├── train_labels.csv    # 格式: id,label
├── train/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...

CSV 文件格式示例：

id,label
image1.tif,0
image2.tif,1
image3.tif,0

### 2. 运行训练

```bash
python main_train.py
```

### 3. 训练参数配置

在 `main_train.py` 中修改以下参数：

```python
batch_size = 64         # 批次大小
num_epochs = 5          # 训练轮数
lr = 1e-3               # 学习率
pretrained = True       # 是否使用预训练权重
```

## 关键配置详解

### 类别权重

```python
class_weight = torch.tensor([1.0, 4.0], device=device)
```

- 第一类权重: 1.0
- 第二类权重: 4.0（多数类/少数类的比例，需根据实际数据调整）

### 数据增强

`build_transforms()` 函数在训练时应用数据增强：

- 训练时：随机裁剪、翻转等
- 验证时：仅标准化处理

## 运行示例

device: cuda
TRAIN total= 1600 dist= {0: (800, 0.5), 1: (800, 0.5)}
VAL total= 400 dist= {0: (200, 0.5), 1: (200, 0.5)}
  Batch 10: loss 0.5432, acc 0.7813
  Batch 20: loss 0.4821, acc 0.8125
  ...

## 注意事项

1. **类别权重**：需根据实际数据集的类别比例调整
2. **学习率**：迁移学习通常使用较小的学习率（1e-4 ~ 1e-3）
3. **冻结策略**：可在第 N 个 epoch 后解冻主干网络进行微调
4. **数据路径**：确保 CSV 文件中的图像名称与实际文件匹配（包括后缀）

## 扩展建议

- [ ] 添加学习率调度器（LR Scheduler）
- [ ] 实现 Early Stopping 机制
- [ ] 保存最优模型权重
- [ ] 集成 TensorBoard 可视化
- [ ] 支持模型推理脚本
- [ ] 添加测试集评估

## 许可证

MIT
