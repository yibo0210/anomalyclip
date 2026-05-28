# AnomalyCLIP ViT-B/16 模型改造计划书

**日期**: 2026-05-26 (更新: 2026-05-28) | **基线模型**: AnomalyCLIP (ICLR 2024) | **主干**: ViT-B/16 | **GPU**: RTX 3060 Laptop 6GB

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
| 多尺度聚合 | AvgPool 1/3/5 on similarity map | 无效 (AUPRO +0.1) | 平均池化模糊边界，无帮助 |
| 分辨率提升 | 224→336, batch 16→8 | **部分有效 (AUPRO +1.5)** | 更细 patch 网格有效，但单分辨率提升有限 |
| Attention Adapter | Self-attn + MLP on patch tokens | 无效 (AUPRO -0.1) | 见下方根因分析 |

### 1.3 根因分析

ViT-B/16 相比 ViT-L/14 有三个结构性劣势：

1. **层数减半**（12 vs 24）→ 多尺度特征层次不够丰富
2. **特征维度减小**（768 vs 1024）→ 表达能力下降
3. **patch 网格更粗**（14×14 vs 24×24）→ 空间定位精度不足

**核心瓶颈**：pixel AUPRO（像素级异常定位）差距最大（-9.9），说明 ViT 缺乏 CNN 式的局部空间上下文感知——14×14 的 patch 网格无法有效表征划痕、裂纹等小尺寸缺陷的空间结构。

### 1.4 Adapter 实验失败的深层原因

Phase 3 的 AttentionAdapter 实验揭示了一个关键架构问题：

**视觉编码器完全冻结**：`VisionTransformer.forward()` 中 `torch.no_grad()` 包裹了整个 transformer 前向传播（`AnomalyCLIP.py:398`），adapter 虽然在 no_grad 块之外，能接收梯度，但其**输入是 detached 的特征**——adapter 只能变换已冻结的特征，无法改变 ViT 提取什么特征。

**结论**：在 frozen ViT 上叠加 adapter 是"治标不治本"。要真正提升特征质量，必须让 ViT 特征本身适应异常模式，或者在不修改 ViT 特征的情况下，从后处理/推理策略上寻找突破。

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

## 三、实验计划（原始）

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

---

## 三-续、Phase 1-3 实验结果

### Phase 1: Multi-Scale Spatial Aggregation ❌ 无效

**实验结果** (2026-05-27, epoch 10, MVTec AD):

| 指标 | Baseline | Phase 1 | Delta |
|------|:--------:|:-------:|:-----:|
| pixel_auroc | 94.1 | 93.5 | -0.6 |
| pixel_aupro | 84.0 | 84.1 | **+0.1** |
| image_auroc | 90.4 | 90.1 | -0.3 |
| image_ap | 94.8 | 94.6 | -0.2 |

**结论**: 无效。平均池化模糊了 anomaly map 边界。

### Phase 2: 分辨率提升 △ 部分有效

**实验结果 (epoch 15)** (2026-05-27, MVTec AD):

| 指标 | Baseline | Phase 2 | Delta |
|------|:--------:|:-------:|:-----:|
| pixel_auroc | 94.1 | 93.5 | -0.6 |
| pixel_aupro | 84.0 | 85.2 | **+1.2** |
| image_auroc | 90.4 | 89.5 | -0.9 |
| image_ap | 94.8 | 94.9 | +0.1 |

**补充实验 (epoch 25, 更充分训练)**:

| 指标 | Baseline | Phase 2 (25ep) | Delta |
|------|:--------:|:--------------:|:-----:|
| pixel_auroc | 94.1 | 93.8 | -0.3 |
| pixel_aupro | 84.0 | 85.5 | **+1.5** |
| image_auroc | 90.4 | 89.4 | -1.0 |
| image_ap | 94.8 | 94.9 | +0.1 |

**结论**: AUPRO +1.5（更多训练轮数后提升更大），部分有效但仍未达 +3 判决标准。

### Phase 3: Attention Adapter ❌ 无效

**实验结果** (2026-05-27, epoch 15, MVTec AD):

| 指标 | Baseline | Phase 3 | Delta |
|------|:--------:|:-------:|:-----:|
| pixel_auroc | 94.1 | 94.1 | 0 |
| pixel_aupro | 84.0 | 83.9 | -0.1 |
| image_auroc | 90.4 | 89.6 | -0.8 |
| image_ap | 94.8 | 94.5 | -0.3 |

**结论**: 无效。adapter 输入是 frozen ViT 的 detached 特征，无法改变 ViT 提取什么特征。

---

## 四、新实验计划（2026-05-28 更新）

基于 Phase 1-3 的经验教训，转向**不依赖 ViT 梯度**的方向——通过推理策略和后处理提升性能。

### Phase 4: Multi-Resolution Test-Time Augmentation (MR-TTA) ★★★★★

**来源**: 基于 Phase 2 分辨率提升的部分有效性 (+1.5 AUPRO)，扩展为多分辨率融合策略

**原理**: 在多个分辨率 (224/280/336) 下分别推理，生成多个 anomaly map，然后融合。小缺陷在高分辨率下更清晰（更细 patch 网格），大缺陷在低分辨率下更有上下文。关键创新是**学习最优融合权重**而非简单平均。

**优势**:
- 零可训练参数（或极少量权重参数）
- 不修改 ViT 结构，不依赖梯度
- 直接利用已验证有效的分辨率提升思路
- 推理时顺序执行各分辨率，峰值显存 ~3GB

**实现**:
1. 修改 `test.py`，在 224/280/336 三个分辨率下分别推理
2. 将各分辨率的 anomaly map 上采样到统一尺寸
3. 融合策略对比：
   - E1: 简单平均
   - E2: 按分辨率加权（可学习标量权重，训练时优化）
   - E3: 逐像素最大值

**实验设计**:

| 实验 | 分辨率组合 | 融合方式 | 预期 AUPRO |
|------|-----------|---------|:---------:|
| E1 | 224+280+336 | 平均 | +2~3 |
| E2 | 224+280+336 | 加权 | +2~4 |
| E3 | 224+280+336 | 逐像素max | +2~4 |
| E4 | 224+336 | 平均（基线对照） | +1~2 |

**判决标准**: AUPRO ≥ +2 即视为有效，≥ +3 即显著

**预计时间**: 0.5 天

### Phase 5: Cross-Scale Attention Fusion ★★★★

**来源**: FPN/BiFPN 多尺度特征融合思路，应用于 anomaly map 层间融合

**原理**: 替代当前 `anomaly_map.sum(dim=0)` 的简单求和，用轻量 cross-attention 让深层（语义）特征引导浅层（空间）特征的 anomaly map 计算。deep features 作为 query，shallow features 作为 key/value，在 patch 级别进行跨层信息交换。

**与已失败方法的区别**:
- FeatureFusionModule 用全局池化 → 丢失空间信息 ❌
- AttentionAdapter 用单层 self-attention → 无法跨层 ❌
- Cross-Scale Attention 用**层间 cross-attention + 保留空间维度** ✅

**优势**:
- 参数极少（~10K），显存可忽略
- 直接改进层间信息融合方式
- 保留 patch 级别的空间信息

**实现**:
1. 在 `test.py` 的 anomaly map 计算部分，提取各层 patch 特征后
2. 计算 cross-attention: Q=deep_features, K/V=shallow_features
3. 用 attention-weighted 特征替代简单求和

**实验设计**:

| 实验 | 注意力类型 | 参数量 | 预期 AUPRO |
|------|-----------|:------:|:---------:|
| F1 | 12→3 cross-attn | ~6K | +1~3 |
| F2 | 12→9→6→3 逐层引导 | ~10K | +2~4 |
| F3 | 双向（top-down + bottom-up） | ~20K | +2~4 |

**判决标准**: AUPRO ≥ +2 即视为有效

**预计时间**: 1 天

### Phase 6: Test-Time Feature Statistics Adaptation ★★★

**来源**: PatchCore 的特征分布建模思路 + 测试时自适应

**原理**: 对每张测试图像，用其自身 patch 特征的统计分布（均值/标准差）识别异常 patch——异常 patch 在特征空间中是离群点。零训练，零额外参数。

**实现**:
1. 提取各层 patch 特征
2. 计算每层 patch 特征的 mean 和 std
3. 异常分数 = 每个 patch 与层统计量的马氏距离
4. 将统计异常分数与 CLIP 相似度 anomaly map 融合

**实验设计**:

| 实验 | 融合方式 | 预期 AUPRO |
|------|---------|:---------:|
| G1 | 乘法融合 | +1~2 |
| G2 | 加权平均 | +1~2 |
| G3 | 逐像素 max | +1~3 |

**判决标准**: AUPRO ≥ +2 即视为有效

**预计时间**: 0.5 天

### Phase 7: 组合最优方案

| 实验 | 组合 | 预期 vs baseline |
|------|------|:---:|
| H1 | Phase 4_best + Phase 2(336) | AUPRO +3~5 |
| H2 | Phase 4_best + Phase 5_best | AUPRO +3~6 |
| H3 | Phase 4_best + Phase 5_best + Phase 6 | AUPRO +4~8 |
| H4 | 全部组合 | AUPRO +5~10 |

---

## 五、判决标准与决策节点

### 判决标准

| 指标 | 有效 | 显著 | 论文可用 |
|------|:---:|:---:|:---:|
| pixel AUPRO Δ | ≥ +2 | ≥ +5 | ≥ +7 |
| pixel AUROC Δ | ≥ +1 | ≥ +2 | ≥ +3 |
| image AUROC Δ | ≥ +1 | ≥ +2 | ≥ +3 |

### 决策节点

```
Phase 4 (MR-TTA) ──→ AUPRO +2+? ──是──→ Phase 5 (Cross-Scale Attention)
         │                                      │
         否                                      ↓
         ↓                              Phase 6 (Feature Statistics)
    调整分辨率组合或融合策略                        │
                                                   ↓
                                            Phase 7 (组合最优)
```

---

## 六、预期里程碑

| 阶段 | 时间 | 目标 | 判决 |
|------|:---:|------|:---:|
| Phase 4 (MR-TTA) | 0.5 天 | AUPRO 84→86+ | 是否有效 |
| Phase 5 (Cross-Scale) | 1 天 | AUPRO 86→88+ | 是否有效 |
| Phase 6 (Statistics) | 0.5 天 | AUPRO 88→90+ | 是否有效 |
| Phase 7 (组合) | 0.5 天 | 最终对比表 | 论文核心实验 |

---

## 七、参考文献

1. Fang et al. "AF-CLIP: Zero-Shot Anomaly Detection via Anomaly-Focused CLIP Adaptation." ACM MM 2025.
2. Ma et al. "AA-CLIP: Enhancing Zero-shot Anomaly Detection via Anomaly-Aware CLIP." CVPR 2025.
3. Cheng et al. "MGVCLIP: Multi-Scale Convolution Adapter + Guided Context Optimization." Measurement 2025.
4. Kong et al. "KAnoCLIP." ICASSP 2025.
5. HeadCLIP, arXiv 2025.
6. ViP²-CLIP, arXiv 2025.
7. Gu et al. "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection." ICLR 2024.
8. Roth et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022. (PatchCore)
9. Tan et al. "EfficientDet: Scalable and Efficient Object Detection." CVPR 2020. (BiFPN)
10. Lin et al. "Feature Pyramid Networks for Object Detection." CVPR 2017. (FPN)
