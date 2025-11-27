# -*- coding: utf-8 -*-
"""
使用 BiLSTM 和 BiGRU 对 data_model_141.xlsx 进行
感染(0) vs 脓毒症(1) 二分类，并在测试集上评估和集成，
同时导出测试集上每个样本的预测概率。
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

DATA_PATH = r"D:\PycharmProjects\判定脓毒症的蛋白检测\data_model_141.xlsx"
LABEL_COL = "sepsis_group"
ID_COL = "eid"   # 用作ID，不作为特征

RANDOM_SEED = 42
BATCH_SIZE = 128
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10   # AUC 连续多少 epoch 不提升就早停

BASE_DIR = os.path.dirname(DATA_PATH)
CLASSIC_PROB_PATH = os.path.join(BASE_DIR, "classic_models_test_probs_141.xlsx")
DEEP_PROB_PATH = os.path.join(BASE_DIR, "deep_models_test_probs_141.xlsx")
ALL_PROB_PATH = os.path.join(BASE_DIR, "all_models_test_probs_141.xlsx")


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
    # 1) 读 Excel
    df = pd.read_excel(path)

    print(f"数据形状： {df.shape}")
    print("列名预览：", list(df.columns[:10]), " ...")

    # 2) 标签列检查
    if label_col not in df.columns:
        cand = [c for c in df.columns if "sepsis" in c.lower()]
        raise ValueError(f"找不到标签列 {label_col}，候选列：{cand}")

    print(f"✅ 使用的标签列： {label_col}")

    # 3) 特征 & 标签 & ID
    feature_cols = [c for c in df.columns if c not in [label_col, id_col]]
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(int)
    ids = df[id_col].values

    print(f"特征数量： {X.shape[1]}")
    print("标签分布：")
    print(pd.Series(y).value_counts())

    # 4) 训练 / 测试 划分（保持与传统模型脚本一致：同样的 random_state 和 test_size）
    X_trainval, X_test, y_trainval, y_test, id_trainval, id_test = train_test_split(
        X, y, ids,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # 再从 trainval 中切出验证集
    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
        X_trainval, y_trainval, id_trainval,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_trainval,
    )

    print(
        f"训练集大小： ({X_train.shape[0]}, {X_train.shape[1]})   "
        f"验证集大小： ({X_val.shape[0]}, {X_val.shape[1]})   "
        f"测试集大小： ({X_test.shape[0]}, {X_test.shape[1]})"
    )

    # 5) 计算 pos_weight
    num_pos = (y_train == 1).sum()
    num_neg = (y_train == 0).sum()
    pos_weight_value = num_neg / max(num_pos, 1)
    print(f"训练集中阳性={num_pos}, 阴性={num_neg}, pos_weight={pos_weight_value:.2f}")

    # 6) 缺失值填补 + 标准化（只用训练集拟合）
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

    pos_weight_tensor = torch.tensor(pos_weight_value, dtype=torch.float32, device=DEVICE)

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        pos_weight_tensor,
        id_test,   # 把测试集的 eid 一起返回
    )


# ==================== Dataset 定义 ====================

class ProteinDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)  # (N, 141)
        self.y = torch.tensor(y, dtype=torch.float32)  # (N,)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        seq = self.X[idx]        # (141,)
        label = self.y[idx]      # 标量
        return seq, label


# ==================== 模型定义：BiLSTM 和 BiGRU ====================

class BiLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=1, dropout=0.3):
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
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        out, _ = self.lstm(x)  # out: (batch, seq_len, 2*hidden)
        last = out[:, -1, :]   # (batch, 2*hidden)
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
        x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        out, _ = self.gru(x)  # (batch, seq_len, 2*hidden)
        last = out[:, -1, :]
        logits = self.fc(last).squeeze(-1)
        return logits


# ==================== 评估函数（训练 & 测试公用） ====================

def evaluate(model, dataloader, name="Model", threshold=0.5, verbose=True):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            logits = model(X_batch)  # (batch,)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(y_batch.detach().cpu().numpy())

    y_prob = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # 保险：把 NaN / Inf 清掉，避免 roc_auc_score 报错
    y_prob = np.nan_to_num(y_prob, nan=0.5, posinf=1.0, neginf=0.0)

    # AUC 需要正负类都存在，否则报错
    try:
        if len(np.unique(y_true)) < 2:
            val_auc = 0.5
        else:
            val_auc = roc_auc_score(y_true, y_prob)
    except Exception as e:
        print(f"[{name}] 计算 AUC 出错：{e}，将 AUC 置为 0.5")
        val_auc = 0.5

    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)

    if verbose:
        print(f"[{name}] AUC = {val_auc:.4f}, ACC = {acc:.4f}")
        cm = confusion_matrix(y_true, y_pred)
        print("混淆矩阵：")
        print(cm)
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
            logits = model(X_batch)  # (batch,)
            loss = criterion(logits, y_batch)

            if torch.isnan(loss):
                print(f"[{model_name}] 第 {epoch} 个 epoch 出现 NaN loss，跳过该 batch")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)

        avg_loss = running_loss / len(train_loader.dataset)

        # 在验证集上评估（不打印混淆矩阵，避免太多输出）
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
        id_test,
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

    # 3) BiLSTM
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

    # 4) BiGRU
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

    try:
        if len(np.unique(y_test_true)) < 2:
            ens_auc = 0.5
        else:
            ens_auc = roc_auc_score(y_test_true, ens_probs_eq)
    except Exception as e:
        print(f"[Ensemble] 计算 AUC 出错：{e}，将 AUC 置为 0.5")
        ens_auc = 0.5

    ens_pred = (ens_probs_eq >= 0.5).astype(int)
    ens_acc = accuracy_score(y_test_true, ens_pred)

    print(f"Equal-Weighted Ensemble Test AUC = {ens_auc:.4f}, ACC = {ens_acc:.4f}")
    cm = confusion_matrix(y_test_true, ens_pred)
    print("等权 Ensemble 混淆矩阵：")
    print(cm)
    print("等权 Ensemble 分类报告：")
    print(classification_report(y_test_true, ens_pred, digits=4))

    print("\n=====================================================================\n")

    # 6) 导出深度模型在测试集上的预测概率
    df_deep = pd.DataFrame({
        "eid": id_test,
        "label": y_test_true.astype(int),
        "prob_BiLSTM": y_test_prob_lstm.ravel(),
        "prob_BiGRU": y_test_prob_gru.ravel(),
        "prob_Ensemble": ens_probs_eq.ravel(),
    })

    df_deep = df_deep.sort_values(by="eid").reset_index(drop=True)
    df_deep.to_excel(DEEP_PROB_PATH, index=False)
    print(f"✅ 深度模型预测概率已保存到: {DEEP_PROB_PATH}")

    # 7) 如有传统模型概率文件，则合并生成总表
    if os.path.exists(CLASSIC_PROB_PATH):
        df_classic = pd.read_excel(CLASSIC_PROB_PATH)
        df_all = pd.merge(df_classic, df_deep, on=["eid", "label"], how="inner")
        df_all.to_excel(ALL_PROB_PATH, index=False)
        print(f"✅ 与传统模型概率已合并，保存到: {ALL_PROB_PATH}")
        print(f"合并后样本数: {df_all.shape[0]}")
    else:
        print(f"⚠ 未找到传统模型概率文件: {CLASSIC_PROB_PATH}，仅保存了深度模型概率。")


if __name__ == "__main__":
    main()
