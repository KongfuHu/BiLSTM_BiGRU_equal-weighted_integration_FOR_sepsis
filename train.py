# -*- coding: utf-8 -*-
"""
使用 BiLSTM 和 BiGRU 对 data_ukb.xlsx 进行
感染(0) vs 脓毒症(1) 二分类，并在测试集上评估和集成。

本脚本包括：
- 数据加载与划分（train/val/test）
- 简单的缺失值填补与标准化
- BiLSTM / BiGRU 模型结构定义
- 训练与早停
- 在验证集和测试集上的性能评估
"""

import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==================== 全局配置 ====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备: {DEVICE}")

# 数据路径与列名（开源时请在 README 中说明数据格式）
DATA_PATH = r"D:\PycharmProjects\判定脓毒症的蛋白检测\data_ukb.xlsx"
LABEL_COL = "sepsis_group"   # 标签列：0=感染, 1=脓毒症
ID_COL = "eid"               # 用作ID，不作为特征（如果没有可以删掉此列名）

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10     # 验证集 AUC 连续多少 epoch 不提升就早停


# ==================== 工具函数：设定随机种子 ====================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(RANDOM_SEED)


# ==================== 数据加载与划分 ====================

def load_and_split_data(
    path,
    label_col=LABEL_COL,
    id_col=ID_COL,
    test_size=0.2,
    val_size=0.2,
    random_state=RANDOM_SEED,
):
    """读取 Excel，完成 train/val/test 划分，并做缺失值填补和标准化。"""
    df = pd.read_excel(path)

    print(f"数据形状： {df.shape}")
    print("列名预览：", list(df.columns[:10]), " ...")

    # 标签列检查
    if label_col not in df.columns:
        cand = [c for c in df.columns if "sepsis" in c.lower()]
        raise ValueError(f"找不到标签列 {label_col}，候选列：{cand}")
    print(f"✅ 使用的标签列： {label_col}")

    # 特征列：去掉标签列和ID列（如果存在）
    drop_cols = [label_col]
    if id_col in df.columns:
        drop_cols.append(id_col)

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)

    print(f"特征数量： {X.shape[1]}")
    print("标签分布：")
    print(pd.Series(y).value_counts())

    # 训练 / 测试 划分
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 再从 trainval 中切出验证集
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_trainval,
    )

    print(
        f"训练集大小： ({X_train.shape[0]}, {X_train.shape[1]})   "
        f"验证集大小： ({X_val.shape[0]}, {X_val.shape[1]})   "
        f"测试集大小： ({X_test.shape[0]}, {X_test.shape[1]})"
    )

    # 计算 pos_weight（用于处理类别不平衡）
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight_value = num_neg / max(num_pos, 1)
    print(f"训练集中阳性={num_pos}, 阴性={num_neg}, pos_weight={pos_weight_value:.2f}")

    # 缺失值填补 + 标准化（拟合只基于训练集）
    imputer = SimpleImputer(strategy="mean")
    scaler = StandardScaler()

    X_train = imputer.fit_transform(X_train)
    X_train = scaler.fit_transform(X_train)

    X_val = imputer.transform(X_val)
    X_val = scaler.transform(X_val)

    X_test = imputer.transform(X_test)
    X_test = scaler.transform(X_test)

    # 转成 float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    pos_weight_tensor = torch.tensor(
        pos_weight_value, dtype=torch.float32, device=DEVICE
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        pos_weight_tensor,
    )


# ==================== Dataset 定义 ====================

class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, num_features)
        self.y = torch.tensor(y, dtype=torch.float32)  # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        seq = self.X[idx]        # (seq_len,) 这里把每个样本视为一条“序列”
        label = self.y[idx]      # 标量
        return seq, label


# ==================== 模型定义：BiLSTM 和 BiGRU ====================

class BiLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, dropout=0.3):
        """
        input_dim: 每个时间步的特征维度，这里把 141 个蛋白视为长度=141 的序列，每步1维
        hidden_dim: LSTM 隐层大小
        num_layers: 堆叠的 LSTM 层数
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len)  →  (batch, seq_len, 1)
        x = x.unsqueeze(-1)
        out, _ = self.lstm(x)           # (batch, seq_len, 2*hidden)
        last = out[:, -1, :]            # 取最后一个时间步
        logits = self.fc(last).squeeze(-1)  # (batch,)
        return logits


class BiGRU(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.unsqueeze(-1)             # (batch, seq_len, 1)
        out, _ = self.gru(x)            # (batch, seq_len, 2*hidden)
        last = out[:, -1, :]
        logits = self.fc(last).squeeze(-1)
        return logits


# ==================== 评估函数 ====================

def evaluate(model, dataloader, name="Model", threshold=0.5, verbose=True):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)          # (batch,)
            probs = torch.sigmoid(logits)    # (batch,)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # 保险：把 NaN / Inf 清掉
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    # AUC 需要正负类都存在
    if len(np.unique(y_true)) < 2:
        val_auc = 0.5
    else:
        val_auc = roc_auc_score(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"[{name}] AUC = {val_auc:.4f}, ACC = {acc:.4f}")
        print("混淆矩阵：")
        print(confusion_matrix(y_true, y_pred))
        print("分类报告：")
        print(classification_report(y_true, y_pred, digits=4))

    return val_auc, acc, y_true, y_prob, y_pred


# ==================== 训练函数 ====================

def train_model(
    model,
    train_loader,
    val_loader,
    pos_weight,
    num_epochs=EPOCHS,
    model_name="Model",
):
    """标准训练循环 + 验证集 AUC 早停。"""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_auc = -1.0
    best_state_dict = None
    no_improve_epochs = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)       # (batch,)
            loss = criterion(logits, y_batch)

            if torch.isnan(loss):
                print(f"[{model_name}] 第 {epoch} 个 epoch 出现 NaN loss，跳过该 batch")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # 在验证集上评估
        val_auc, val_acc, _, _, _ = evaluate(
            model, val_loader, name=model_name, verbose=False
        )

        print(
            f"[{model_name}] Epoch [{epoch}/{num_epochs}] "
            f"Train Loss = {avg_loss:.4f} | Val AUC = {val_auc:.4f} | Val ACC = {val_acc:.4f}"
        )

        # 早停逻辑：Val AUC 提升就保存
        if val_auc > best_auc:
            best_auc = val_auc
            best_state_dict = model.state_dict()
            no_improve_epochs = 0
            print(f"  🔥 [{model_name}] Val AUC 提升为 {val_auc:.4f}，保存当前模型")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(
                    f"  ❗[{model_name}] Val AUC 连续 {EARLY_STOP_PATIENCE} 个 epoch 未提升，提前停止"
                )
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    print(f"\n================= {model_name} 在验证集上的最终表现 =================")
    evaluate(model, val_loader, name=model_name + " Final Val", verbose=True)

    return model, best_auc


# ==================== 主函数 ====================

def main():
    # 1) 加载数据
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        pos_weight,
    ) = load_and_split_data(DATA_PATH)

    # 2) 构造 Dataset / DataLoader
    train_dataset = ProteinDataset(X_train, y_train)
    val_dataset = ProteinDataset(X_val, y_val)
    test_dataset = ProteinDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    # 3) 训练 BiLSTM
    bilstm = BiLSTM(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.3).to(DEVICE)
    print(bilstm)
    print("\n================= 开始训练模型： BiLSTM =================")
    bilstm, best_auc_lstm = train_model(
        bilstm, train_loader, val_loader, pos_weight, num_epochs=EPOCHS, model_name="BiLSTM"
    )

    print("\n================= BiLSTM 在测试集上的表现 =================")
    _, _, y_test_true, y_test_prob_lstm, _ = evaluate(
        bilstm, test_loader, name="BiLSTM Test", verbose=True
    )

    # 4) 训练 BiGRU
    bigru = BiGRU(input_dim=1, hidden_dim=64, num_layers=1, dropout=0.3).to(DEVICE)
    print(bigru)
    print("\n================= 开始训练模型： BiGRU =================")
    bigru, best_auc_gru = train_model(
        bigru, train_loader, val_loader, pos_weight, num_epochs=EPOCHS, model_name="BiGRU"
    )

    print("\n================= BiGRU 在测试集上的表现 =================")
    _, _, _, y_test_prob_gru, _ = evaluate(
        bigru, test_loader, name="BiGRU Test", verbose=True
    )

    # 5) 简单集成：BiLSTM + BiGRU 概率平均
    print("\n================= 多模型集成（BiLSTM + BiGRU，等权概率平均） =================")
    ens_probs_eq = (y_test_prob_lstm + y_test_prob_gru) / 2.0
    ens_probs_eq = np.nan_to_num(ens_probs_eq, nan=0.5, posinf=1.0, neginf=0.0)

    if len(np.unique(y_test_true)) < 2:
        ens_auc = 0.5
    else:
        ens_auc = roc_auc_score(y_test_true, ens_probs_eq)

    ens_pred = (ens_probs_eq >= 0.5).astype(int)
    ens_acc = accuracy_score(y_test_true, ens_pred)

    print(f"Equal-Weighted Ensemble Test AUC = {ens_auc:.4f}, ACC = {ens_acc:.4f}")
    print("等权 Ensemble 混淆矩阵：")
    print(confusion_matrix(y_test_true, ens_pred))
    print("等权 Ensemble 分类报告：")
    print(classification_report(y_test_true, ens_pred, digits=4))


if __name__ == "__main__":
    main()
