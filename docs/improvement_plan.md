# AnomalyCLIP ViT-B/16 模型改造计划书

**日期**: 2026-05-26 | **基线模型**: AnomalyCLIP (ICLR 2024) | **主干**: ViT-B/16 | **GPU**: RTX 3060 Laptop 6GB

---

## 一、问题定义

### 1.1 当前基线性能（MVTec AD）

| 指标 | ViT-B/16 (ours) | ViT-L/14 (原论文) | 差距 |
|------|:------:|:------:|:----:|
| pixel_auroc | 94.1 | 97.8 | -3.7 |
| pixel_aupro | 84.0 | 93.9 | **-9.9** |
| image_auroc | 90.4 | 95.6 | -5.2 |

### 1.2 已排除的方向

| 方向 | 方法 | 结果 | 原因分析 |
|------|------|:---:|------|
| 特征融合 | FeatureFusionModule (300 params) | 无效 | 全局池化丢失空间信息 |
| 边界优化 | BoundaryLoss + LearnableRefine | 无效 | 训练集无 GT mask |
| PEFT - Adapter | BottleneckAdapter @ FFN (0.3M) | 无效 | 改 FFN 特征变换不改注意力焦点 |
| PEFT - LoRA | AttentionLoRA rank=4 (37K) | 无效 | 同上 |

### 1.3 根因分析

ViT-B/16 相比 ViT-L/14 有三个结构性劣势：

1. **层数减半**（12 vs 24）→ 多尺度特征层次不够丰富
2. **特征维度减小**（768 vs 1024）→ 表达能力下降
3. **patch 网格更粗**（14×14 vs 24×24）→ 空间定位精度不足

**核心瓶颈**：pixel AUPRO（像素级异常定位）差距最大（-9.9），说明 ViT 缺乏 CNN 式的局部空间上下文感知——14×14 的 patch 网格无法有效表征划痕、裂纹等小尺寸缺陷的空间结构。

---

## 二、改进方向

基于 2024-2025 年前沿论文调研（AF-CLIP / AA-CLIP / MGVCLIP / HeadCLIP / ViP²-CLIP / AULoRA），筛选出三个可行方向。

### 方向 A：Multi-Scale Spatial Aggregation ★★★★★

**来源**: AF-CLIP (Fang et al., ACM MM 2025)

**原理**: 在计算 anomaly map 之前，对每层 patch 特征做多尺度空间聚合，用 1×1、3×3、5×5 三个滑窗做高斯加权平均并拼接。弥补 ViT 缺乏局部归纳偏置的缺陷。

**优势**:
- 零可训练参数，纯粹的特征预处理
- 直接针对 pixel AUPRO 低的根因（缺乏局部空间上下文）
- 实现简单，可插入现有 AnomalyCLIP pipeline 的 `fused_anomaly_map()` 之前

**参考结果**: MVTec AD 零样本 pixel AUROC 92.3%，4-shot 97.6%

### 方向 B：增大输入分辨率 ★★★★

**原理**: 将输入从 224×224 提升至 336×336 或 384×384。patch 数量由 14×14→24×24（336）或 27×27（384）。

**优势**:
- ViT 自带位置编码插值，无需改模型结构
- 更细的空间网格 → 小缺陷不会被一个 patch 吞没
- 零参数开销

**劣势**: 显存增加，batch_size 需从 16 降至 8

**显存估算（336px）**:
- 当前 224px: ~2.8GB（batch=16）
- 336px: ~5.5GB（batch=8），6GB 显存可以跑

### 方向 C：Anomaly-Focused Attention Adapter ★★★★

**来源**: AF-CLIP / AA-CLIP (CVPR 2025)

**原理**: 在 ViT 各层输出后加轻量 cross-attention 模块，用可学习的 Q/K/V 投影重新加权 patch token——让 [CLS] token 学会关注异常区域，让 patch token 放大缺陷特征。

**与你之前 Adapter 的关键区别**:

| | 旧 Adapter | 新 Attention Adapter |
|------|------|------|
| 插入位置 | FFN 后 | Attention 输出后 |
| 机制 | MLP 特征变换 | Self-attention 重加权 |
| 改什么 | 特征值 | 特征间关系 |
| 对症 | 否 | 是 |

**参数量**: ~0.5% 总参数（约 0.4M）

**参考结果**: AA-CLIP 在 MVTec AD 上 full-shot pixel AUROC 93.4%（SOTA）

---

## 三、实验计划

### Phase 1: Multi-Scale Spatial Aggregation（优先级最高）

**目标**: 验证空间聚合能否提升 AUPRO

**实现**:
1. 在 `prompt_ensemble.py` 的 `fused_anomaly_map()` 中，对每层 patch feature 做多尺度高斯滑窗聚合
2. 滑窗尺寸: r ∈ {1, 3, 5}，高斯 σ = r/3
3. 聚合特征与原始特征拼接过 attention adapter（或用简单线性投影降维）

**实验设计**:

| 实验 | 滑窗尺寸 | 聚合方式 | 预期 AUPRO 变化 |
|------|:---:|------|:---:|
| A1 | 1,3,5 | 拼接+线性投影 | +3~5 |
| A2 | 1,3,5 | 平均池化 | +2~4 |
| A3 | 1,3,5,7 | 拼接+线性投影 | +3~6 |

**判决标准**: AUPRO 提升 ≥ 3 点即视为有效

### Phase 2: 分辨率提升（与 Phase 1 并行或接续）

**目标**: 验证更高分辨率能否提升 AUPRO

**实现**:
1. 修改 train.py / test.py 的 `--image_size` 从 224 → 336
2. batch_size 降至 8
3. 位置编码自动插值（ViT 已支持）

**实验设计**:

| 实验 | 分辨率 | batch_size | 预期 |
|------|:---:|:---:|------|
| B1 | 336×336 | 8 | AUPRO +3~7 |
| B2 | 384×384 | 4 | AUPRO +5~10（但显存紧张） |

### Phase 3: Attention Adapter（前两个方向验证后）

**目标**: 在最优预处理基础上加轻量注意力适配

**实现**:
1. 每层添加 `Q/K/V` 投影矩阵（维度 d×d̃，d̃=64）
2. 输出投影 `W_O`
3. 残差连接: `x' = x + Attention(Qx, Kx, Vx)`

**实验设计**:

| 实验 | 前置条件 | adapter 层数 | 预期 |
|------|------|:---:|------|
| C1 | Phase 1 最优方案 | 6-12层 | +1~3 |
| C2 | Phase 1+2 最优方案 | 6-12层 | +1~3 |

### Phase 4: 组合最优方案

| 实验 | 组合 | 预期 vs baseline |
|------|------|:---:|
| D1 | A_best + B | AUPRO +5~10 |
| D2 | A_best + C | AUPRO +4~8 |
| D3 | A_best + B + C | AUPRO +7~12 |

---

## 四、判决标准与决策节点

### 判决标准

| 指标 | 有效 | 显著 | 论文可用 |
|------|:---:|:---:|:---:|
| pixel AUPRO Δ | ≥ +2 | ≥ +5 | ≥ +7 |
| pixel AUROC Δ | ≥ +1 | ≥ +2 | ≥ +3 |
| image AUROC Δ | ≥ +1 | ≥ +2 | ≥ +3 |

### 决策节点

```
Phase 1 (Multi-Scale) ──→ AUPRO +3+? ──是──→ Phase 2 (分辨率)
         │                                      │
         否                                      ↓
         ↓                              Phase 3 (Attention Adapter)
    调整滑窗参数                                 │
    或跳过                                        ↓
                                           Phase 4 (组合)
```

---

## 五、预期里程碑

| 阶段 | 时间 | 目标 | 判决 |
|------|:---:|------|:---:|
| Phase 1 | 1 天 | AUPRO 84→87+ | 是否有效 |
| Phase 2 | 0.5 天 | AUPRO 87→90+ | 显存是否够 |
| Phase 3 | 1 天 | AUPRO 90→92+ | 是否叠加有效 |
| Phase 4 | 0.5 天 | 最终对比表 | 论文核心实验 |

---

## 六、参考文献

1. Fang et al. "AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation." ACM MM 2025.
2. Ma et al. "AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP." CVPR 2025.
3. Cheng et al. "MGVCLIP: Multi-Scale Convolution Adapter + Guided Context Optimization." Measurement 2025.
4. Kong et al. "KAnoCLIP." ICASSP 2025.
5. HeadCLIP, arXiv 2025.
6. ViP²-CLIP, arXiv 2025.
7. Gu et al. "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection." ICLR 2024.
