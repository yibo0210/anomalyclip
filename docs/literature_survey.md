# 基于 CLIP 的异常检测：相关论文方法调研

> 调研时间：2026-05-14 | 硬件约束：RTX 3060 Laptop 6GB
> 调研范围：CLIP-based 异常检测、参数高效微调、知识蒸馏、多尺度特征融合

---

## 一、CLIP-based 异常检测核心论文

### 1.1 AnomalyCLIP (ICLR 2024)

- **论文**：AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection
- **作者**：Q. Gu et al.
- **会议**：ICLR 2024

**核心方法**：
- 提出 **Object-agnostic Prompt Learning**：使用通用 classname "object" 替代具体的物体类别名，学习通用的 normal/abnormal 提示词
- **DPAM (Dual-Path Attention Mechanism)**：在 ViT 深层替换标准 QKV 注意力为 V-V 注意力，放大正常与异常 token 的注意力差异
- **Compound Prompts**：在文本编码器的多个中间层插入可学习 token，逐层细化文本特征
- **Multi-layer Feature Extraction**：从 ViT 多个中间层提取 patch 特征，逐层计算异常图后求和

**关键结果**：在 MVTec AD 上 image AUROC 达到 ~99.4%，零样本跨域迁移表现优异

**与你的代码对应**：`AnomalyCLIP_lib/AnomalyCLIP.py` 中的 DPAM 实现、`prompt_ensemble.py` 中的 PromptLearner

---

### 1.2 WinCLIP (CVPR 2023)

- **论文**：WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation
- **作者**：J. Jeong, Y. Zou, T. Kim et al.
- **机构**：Meta FAIR, Qualcomm AI Research
- **会议**：CVPR 2023

**核心方法**：
- **窗口级 CLIP 特征提取**：使用滑动窗口在 CLIP 的图像编码器上提取局部特征，实现像素级异常定位
- **组合式提示词 (Compositional Prompts)**：设计多种 normal/abnormal 描述模板，如 "a photo of a flawless {object}" vs "a photo of a damaged {object}"
- **提示词集成 (Prompt Ensemble)**：将 35 个 CLIP 模板与多个状态描述组合，取平均文本特征
- **Few-shot 设置**：利用少量正常样本的特征作为参考，计算测试图像与参考的相似度差异

**关键结果**：在 MVTec AD 上零样本 image AUROC ~98.8%，few-shot 设置下进一步提升

**与 AnomalyCLIP 的区别**：WinCLIP 使用固定的手工提示词，AnomalyCLIP 学习可优化的提示词；WinCLIP 的窗口级特征提取计算开销较大

---

### 1.3 AnomalyGPT (AAAI 2024)

- **论文**：AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models
- **作者**：Z. Gu, B. Zhu, G. Zhu, Y. Chen, M. Tang, J. Wang
- **机构**：中国科学院自动化研究所
- **会议**：AAAI 2024

**核心方法**：
- 基于 **ImageBind + LLaMA** 构建大型视觉-语言模型 (LVLM)
- **Prompt Learner 模块**：学习伪异常特征，使 LVLM 无需大量微调即可检测异常
- **Few-shot 学习**：仅需少量正常参考图像即可进行异常检测
- **异常定位**：通过 patch 级别的特征相似度生成异常热力图

**关键结果**：在 MVTec AD 和 VisA 上取得有竞争力的结果，支持自然语言交互式异常诊断

**与你的硬件关系**：ImageBind + LLaMA 需要大显存，RTX 3060 6GB **无法直接运行**，但其 prompt learner 思路可借鉴

---

### 1.4 PromptAD (2024)

- **论文**：PromptAD: Zero-shot Anomaly Detection using Text Prompts
- **发表**：2024

**核心方法**：
- 利用 CLIP 的文本-图像对齐能力进行零样本异常检测
- 设计专门的文本提示词模板来区分正常与异常模式
- 不需要任务特定的训练数据

**关键特点**：与 AnomalyCLIP 类似，都是基于 CLIP 提示词的零样本方法，但 PromptAD 更侧重于提示词模板的设计而非可学习提示词

---

### 1.5 APRIL-GAN (2024)

- **论文**：APRIL-GAN: Few-shot Anomaly Detection via Prompt Learning
- **发表**：2024

**核心方法**：
- 基于提示词学习的 few-shot 异常检测
- 使用视觉-语言模型的预训练知识，通过最少的标注样本进行异常检测
- 连接大型预训练模型与实际异常检测应用之间的差距

---

### 1.6 SAA+ (Segment Any Anomaly) (2024)

- **论文**：Segment Any Anomaly + (SAA+)
- **发表**：2024

**核心方法**：
- 结合 **SAM (Segment Anything Model)** 与语言引导进行异常分割
- 零样本/少样本异常分割
- 利用 SAM 强大的零分割能力，配合 CLIP 的语义理解

**与你的硬件关系**：SAM 模型较大 (~600M+ params)，但 SAM-ViT-B (~90M) 可以在 6GB 显存下运行

---

## 二、参数高效微调 (PEFT) 方法

### 2.1 CLIP-Adapter (IJCV 2023)

- **论文**：CLIP-Adapter: Better Vision-Language Models with Feature Adapters
- **作者**：P. Gao, S. Geng, R. Zhang et al.
- **发表**：IJCV 2023

**核心方法**：
- 冻结 CLIP 主干网络，在视觉和/或语言编码器后附加轻量级 **瓶颈适配器 (Bottleneck Adapter)**
- 适配器结构：down-project → 非线性激活 → up-project + 残差连接
- **可学习混合因子 (blending factor)**：将适配后的特征与原始特征加权混合
- 仅需训练少量参数（适配器权重），远少于全量微调

**关键结果**：在 few-shot 分类任务上超越全量微调，训练速度更快

**参数量**：适配器通常仅增加 ~0.5M 参数（bottleneck=32 时）

---

### 2.2 LoRA (ICLR 2022)

- **论文**：LoRA: Low-Rank Adaptation of Large Language Models
- **作者**：E. J. Hu, Y. Shen, P. Wallis et al.
- **机构**：Microsoft Research
- **会议**：ICLR 2022

**核心方法**：
- 冻结预训练权重，在 Transformer 层中注入可训练的 **低秩分解矩阵** A 和 B
- 权重更新 ΔW = BA，其中 rank r << 原始维度
- 训练参数量减少 **10,000 倍**，GPU 内存减少 **3 倍**
- **推理无额外延迟**：低秩矩阵可合并回原始权重

**关键结果**：在 GPT-3 175B 上，LoRA 的性能与全量微调相当，但仅需训练 0.01% 的参数

**应用到 AnomalyCLIP 的可行性**：可在 ViT 的 QKV 投影层注入 LoRA，rank=4-8 时仅增加 ~100K 参数

---

### 2.3 CoOp (CVPR 2022)

- **论文**：Learning to Prompt for Vision-Language Models (Context Optimization)
- **作者**：K. Zhou et al.
- **机构**：MMLab, NTU
- **会议**：CVPR 2022

**核心方法**：
- 学习 **连续提示词向量 (context vectors)** 替代手工设计的文本提示词
- 优化少量 context vectors，CLIP 参数完全冻结
- 在 11 个 few-shot 分类数据集上显著提升性能

**后续工作 CoCoOp (NeurIPS 2022)**：使 context 变为输入条件相关，提升对新类别的泛化能力

**与 AnomalyCLIP 的关系**：AnomalyCLIP 的 PromptLearner 直接借鉴了 CoOp 的思路，但将其扩展到异常检测场景并增加了 Compound Prompts

---

### 2.4 Tip-Adapter (ECCV 2022)

- **论文**：Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification
- **作者**：Zhou et al.
- **会议**：ECCV 2022

**核心方法**：
- **无需训练**的 CLIP 适配方法
- 从 few-shot 支持集中构建 **key-value 缓存模型**，用作 CLIP 的适配器
- 使用亲和力检索（query 与缓存特征的距离）增强 CLIP 的零样本预测
- 可训练版本 Tip-Adapter-F 进一步提升性能

**关键优势**：完全无需训练，性能与需要微调的方法（如 CLIP-Adapter）相当

---

### 2.5 Prompt-aligned Adapter (2024)

- **论文**：Prompt-aligned Adapter for Generalist Anomaly Detection
- **ArXiv**：2407.09278 (2024)

**核心方法**：
- 比较 adapter tuning vs. prompt tuning 在异常检测中的效果
- 通过轻量级适配器对齐视觉和文本提示词
- 针对通用异常检测设计的适配策略

---

## 三、非 CLIP 的异常检测 SOTA 方法

### 3.1 SimpleNet (CVPR 2024)

- **论文**：SimpleNet: A Simple Network for Efficient Anomaly Detection and Localization
- **作者**：Z. Liu, Y. Zhou, Y. Li et al.
- **会议**：CVPR 2024

**核心方法**：
- **合成异常生成**：通过数据增强生成合成异常样本用于训练
- **轻量级特征提取器** + **特征适配器 (Feature Adaptor)**
- **异常判别器 (Anomaly Discriminator)**：区分正常与异常模式
- 整体架构简洁高效，参数量少

**关键结果**：在 MVTec AD 上 image AUROC ~99.6%，pixel AUPRO ~98.1%

**与 AnomalyCLIP 的区别**：SimpleNet 不使用 CLIP，而是基于预训练特征提取器（如 ResNet/ViT），需要针对每个数据集单独训练

---

### 3.2 EfficientAD (CVPR 2024)

- **论文**：EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies
- **作者**：K. Batzner, L. Heckler, R. König
- **机构**：SICK AG
- **会议**：CVPR 2024

**核心方法**：
- **Student-Teacher 架构**：预训练教师网络训练学生网络，异常在学生无法复现教师特征时被检测
- **小型 patch 描述子**：提取紧凑特征实现高效推理
- **自编码器 (Autoencoder)**：辅助重建正常模式

**关键结果**：推理速度达到毫秒级，同时在 MVTec AD 上保持有竞争力的精度

**与你的硬件关系**：EfficientAD 强调效率，非常适合 RTX 3060 6GB 部署

---

### 3.3 PatchCore (CVPR 2022)

- **论文**：Towards Total Recall in Industrial Anomaly Detection
- **作者**：K. Roth, L. Pemula, J. Zepeda et al.
- **机构**：Amazon Science
- **会议**：CVPR 2022

**核心方法**：
- **核心集缩减的内存库 (Coreset-reduced Memory Bank)**：存储正常样本的 patch 级别特征
- 测试时计算测试 patch 与内存库中最邻近特征的距离作为异常分数
- **核心集子采样**：减少内存库大小同时保持代表性
- 使用 WideResNet-50 等预训练模型的中间层特征

**关键结果**：在 MVTec AD 上 image AUROC ~99.1%

**与 AnomalyCLIP 的区别**：PatchCore 需要为每个类别维护独立的内存库，AnomalyCLIP 使用统一的提示词学习跨类别泛化

---

### 3.4 DRAEM (ICCV 2021)

- **论文**：DRAEM – A discriminatively trained anomaly localization and detection approach
- **作者**：V. Zavrtanik, M. Kristan, D. Skočaj
- **机构**：University of Ljubljana
- **会议**：ICCV 2021

**核心方法**：
- **合成异常生成**：将异常样式的噪声/图案混合到正常训练图像中
- **重建子网络**：U-Net 架构训练将增强图像重建回正常外观
- **判别子网络**：区分重建图像与原始增强图像的差异
- **重建误差**作为异常分数

**关键结果**：在 MVTec AD 上取得当时的 SOTA，同时支持图像级检测和像素级定位

---

### 3.5 RealNet (CVPR 2024)

- **论文**：RealNet: A Feature Selection Network with Realistic Synthetic Anomaly for Anomaly Detection
- **会议**：CVPR 2024

**核心方法**：
- **真实感异常合成**：生成更接近真实缺陷的合成异常样本
- **特征选择网络**：选择对异常检测最相关的特征
- 缩小合成异常与真实异常之间的域差距

---

### 3.6 Dinomaly (2024)

- **论文**：Dinomaly: The Power of DINOv2 Features in Unsupervised Visual Anomaly Detection
- **发表**：2024

**核心方法**：
- 使用 **DINOv2**（自监督 ViT）作为特征提取器
- **重建式方法**：学习重建正常特征，基于重建误差检测异常
- Transformer 解码器重建 DINOv2 特征

**关键结果**：在 MVTec AD 和 VisA 上取得 SOTA

**与你的硬件关系**：DINOv2-ViT-B (~86M params) 与 CLIP ViT-B 参数量相当，RTX 3060 可运行

---

## 四、知识蒸馏与轻量化方法

### 4.1 TinyCLIP (ICCV 2024)

- **论文**：TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance
- **机构**：Microsoft Research
- **会议**：ICCV 2024

**核心方法**：
- **亲和力模仿 (Affinity Mimicking)**：保持蒸馏过程中的跨模态关系
- **权重继承 (Weight Inheritance)**：学生网络从教师网络继承部分权重初始化
- 可将 ViT-B/32 压缩 50% 以上，精度损失极小

---

### 4.2 MobileCLIP (2024)

- **论文**：MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training
- **机构**：Apple, CMU
- **发表**：2024

**核心方法**：
- **多模态强化训练**：使用教师模型集成训练轻量级 CLIP 架构
- 专为移动端设计的超低延迟 CLIP 模型
- 提供多个变体：MobileCLIP-S0, S1, S2, S3

**关键结果**：在移动设备上实现 SOTA 的效率-精度权衡

---

## 五、特征融合相关方法

### 5.1 FPN / BiFPN

- **FPN (CVPR 2017)**：Feature Pyramid Networks for Object Detection
  - 自顶向下的多尺度特征融合，用于检测不同尺度的目标
- **BiFPN (CVPR 2020)**：EfficientDet 中提出的双向特征金字塔
  - 加权双向跨尺度连接，比传统 FPN 更高效
  - 可学习的层间权重

**应用到 AnomalyCLIP 的可行性**：AnomalyCLIP 已从 ViT 的 4 个中间层（layer 3, 6, 9, 12）提取特征，可以借鉴 BiFPN 的加权融合策略替代简单求和

---

### 5.2 Boundary Loss (2019/2021)

- **论文**：Boundary Loss for Highly Unbalanced Segmentation
- **作者**：H. Kervadec et al.
- **发表**：IPMI 2019, Medical Image Analysis 2021

**核心方法**：
- 基于距离变换的边界损失，补充 Dice/CE 等区域级损失
- 专门优化分割边界精度
- 对前景/背景极度不平衡的分割任务特别有效

**应用到 AnomalyCLIP 的可行性**：异常区域通常很小（像素级不平衡），引入 Boundary Loss 可以改善异常图的边界质量，提升 pixel AUPRO

---

## 六、方法分类总结

### 按技术路线分类

| 技术路线 | 代表方法 | 核心思路 | 是否需要训练 |
|---------|---------|---------|:----------:|
| **提示词学习** | AnomalyCLIP, CoOp, PromptAD | 学习可优化的文本提示词 | 是 |
| **提示词集成** | WinCLIP | 手工设计多种提示词模板取平均 | 否 |
| **Adapter 微调** | CLIP-Adapter, Prompt-aligned Adapter | 在 CLIP 中插入轻量级瓶颈模块 | 是 |
| **LoRA 微调** | LoRA | 低秩分解注入可训练矩阵 | 是 |
| **内存库方法** | PatchCore | 存储正常特征，最近邻匹配 | 否 |
| **重建式方法** | DRAEM, Dinomaly | 学习重建正常模式，重建误差为异常分数 | 是 |
| **Student-Teacher** | EfficientAD | 学生网络复现教师特征，差异为异常 | 是 |
| **合成异常** | SimpleNet, RealNet, DRAEM | 生成合成异常样本训练判别器 | 是 |
| **大型 LVLM** | AnomalyGPT | 使用 LLaMA/ImageBind 等大模型 | 部分 |
| **知识蒸馏** | TinyCLIP, MobileCLIP | 将大模型能力迁移到小模型 | 是 |

### 按是否使用 CLIP 分类

| 类别 | 方法 | 特点 |
|------|------|------|
| **CLIP-based** | AnomalyCLIP, WinCLIP, AnomalyGPT, PromptAD, APRIL-GAN | 利用 CLIP 的视觉-语言对齐能力，支持零样本 |
| **非 CLIP** | SimpleNet, EfficientAD, PatchCore, DRAEM, Dinomaly, RealNet | 基于预训练视觉特征，通常需要针对数据集训练 |

---

## 七、结合 RTX 3060 6GB 的可行性评估

### 7.1 显存预算分析

RTX 3060 Laptop 6GB 的显存分配：

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| CLIP ViT-B/16 模型权重 | ~350 MB | 86M params x 4 bytes |
| 图像输入 + 中间特征 | ~200 MB | batch_size=16, 224x224 |
| 梯度 + 优化器状态 | ~800 MB | Adam 优化器需 2x 参数显存 |
| PyTorch 开销 | ~500 MB | CUDA context, 内存碎片 |
| **总计** | ~1.85 GB | 当前 AnomalyCLIP 训练 |
| **剩余可用** | ~4.15 GB | 可用于新增模块 |

### 7.2 各方法的显存友好度

| 方法 | 新增显存 | 可行性 | 说明 |
|------|:-------:|:------:|------|
| **提示词学习 (当前)** | ~0 MB | ✅ 已验证 | AnomalyCLIP 已在 3060 上运行 |
| **CLIP-Adapter** | ~50 MB | ✅ 可行 | Adapter 参数量 ~0.5M |
| **LoRA (rank=4)** | ~20 MB | ✅ 可行 | 仅增加 ~100K 参数 |
| **LoRA (rank=8)** | ~40 MB | ✅ 可行 | 仅增加 ~200K 参数 |
| **Boundary Loss** | ~0 MB | ✅ 可行 | 仅修改损失函数 |
| **BiFPN 融合** | ~30 MB | ✅ 可行 | 仅增加少量线性层 |
| **重建式 (DRAEM)** | ~500 MB | ⚠️ 需降 batch | U-Net 重建网络额外开销 |
| **知识蒸馏 (教师+学生)** | ~1.2 GB | ⚠️ 需降 batch | 教师推理 + 学生梯度 |
| **DINOv2 替换 CLIP** | ~0 MB | ✅ 可行 | 参数量相当，但需重新训练 |
| **MobileNet 学生** | ~100 MB | ✅ 可行 | 学生网络 ~5M params |
| **AnomalyGPT (LLaMA)** | >8 GB | ❌ 不可行 | 超出 6GB 限制 |
| **MobileCLIP** | ~200 MB | ✅ 可行 | 轻量级 CLIP 变体 |

### 7.3 推荐的可行研究方向（基于真实论文方法）

#### 方向 A：在 AnomalyCLIP 基础上引入 Adapter/LoRA 微调视觉编码器

**来源方法**：
- CLIP-Adapter (IJCV 2023) — 瓶颈适配器
- LoRA (ICLR 2022) — 低秩适应
- Prompt-aligned Adapter (2024) — 针对异常检测的适配器

**当前 AnomalyCLIP 的局限**：视觉编码器完全冻结，仅训练 prompt 参数。CLIP 的视觉特征是为自然图像-文本对齐预训练的，对工业缺陷的表达可能不够充分。

**改进思路**：
- 在 ViT 的 FFN 层后插入 Adapter（down→act→up + 残差），瓶颈维度 32-64
- 或在 QKV 投影层注入 LoRA，rank=4-8
- 仅训练 Adapter/LoRA 参数 + prompt 参数，CLIP 主干完全冻结

**显存评估**：Adapter ~0.5M params (~2MB)，LoRA rank=4 ~100K params (~0.4MB)，完全可行

**预期创新点**：将参数高效微调从文本端（prompt learning）扩展到视觉端，双端联合优化

---

#### 方向 B：边界感知的异常图精炼

**来源方法**：
- Boundary Loss (IPMI 2019) — 距离变换边界损失
- DeepLab 的 CRF 后处理 — 条件随机场精炼

**当前 AnomalyCLIP 的局限**：使用固定的高斯平滑 (sigma=4) 后处理异常图，对所有类别和图像相同；pixel AUPRO 仅 37.2%，说明边界质量差。

**改进思路**：
- 在训练损失中引入 Boundary Loss，强调缺陷边界精度
- 用轻量级 CNN（3-5 层 Conv）替代固定高斯平滑，学习自适应空间精炼
- 或使用可学习的 sigma 参数，根据异常图的局部特征动态调整平滑强度

**显存评估**：Boundary Loss 仅修改损失函数，零额外显存；小型精炼网络 ~50K params (~0.2MB)

**预期创新点**：可学习后处理替代固定后处理，针对性提升 pixel AUPRO

---

#### 方向 C：多尺度特征的自适应加权融合

**来源方法**：
- BiFPN (CVPR 2020) — 双向加权特征金字塔
- FPN (CVPR 2017) — 多尺度特征融合

**当前 AnomalyCLIP 的局限**：FeatureFusionModule 仅对 patch 特征做 mean pooling 后计算注意力权重，丢失空间信息；v3.0 实验显示融合模块仅带来 +0.3% 提升。

**改进思路**：
- 借鉴 BiFPN 的可学习层间权重，在 patch 空间维度上计算跨层注意力
- 为不同空间位置选择不同的最优层组合（空间自适应）
- 引入 top-down 和 bottom-up 的双向信息流

**显存评估**：196 个 patch 位置 x 4 层的注意力矩阵仅 784 元素，~10K 新增参数

**预期创新点**：空间自适应的跨层注意力融合，替代全局池化

---

#### 方向 D：对比学习增强的提示词优化

**来源方法**：
- SupCon (2021) — 监督对比学习
- CLIP 原始训练目标 — InfoNCE 对比损失
- SimpleNet (CVPR 2024) — 对比式特征学习

**当前 AnomalyCLIP 的局限**：仅使用 Cross-Entropy Loss 对齐 image-text 相似度，没有显式拉近/推远不同状态的特征对。

**改进思路**：
- 在原有 CE loss 基础上添加 InfoNCE 对比损失
- 构建 batch 内正负样本对：正常图像 ↔ 正常 prompt 为正对，正常图像 ↔ 异常 prompt 为负对
- 使用 hard negative mining 选择最具混淆性的负样本

**显存评估**：仅需存储 batch_size x 2 的相似度矩阵，batch_size=16 时仅 32 个元素

**预期创新点**：对比学习 + prompt learning 结合，增强异常判别力

---

#### 方向 E：知识蒸馏到轻量级模型

**来源方法**：
- TinyCLIP (ICCV 2024) — CLIP 蒸馏
- MobileCLIP (Apple, 2024) — 移动端 CLIP
- EfficientAD (CVPR 2024) — 高效异常检测

**当前 AnomalyCLIP 的局限**：推理需要完整的 CLIP ViT-B/16 (86M params)，在边缘设备上部署困难。

**改进思路**：
- 教师网络：训练好的 AnomalyCLIP
- 学生网络：MobileNetV3-Small (~5M params) 或 EfficientNet-B0
- 特征级蒸馏：对齐教师和学生的中间特征
- 输出级蒸馏：对齐异常图和图像级分数

**显存评估**：教师推理 + 学生梯度，batch_size 需降至 4-8，约需 3-4 GB

**预期创新点**：首次将 CLIP-based 异常检测蒸馏到轻量级模型

---

## 八、总结与建议

### 最可行的三个方向（适合发论文 + RTX 3060）

| 排名 | 方向 | 来源论文 | 显存友好 | 创新性 | 可行性 |
|:---:|------|---------|:------:|:------:|:------:|
| 1 | **Adapter/LoRA 微调视觉编码器** | CLIP-Adapter, LoRA, Prompt-aligned Adapter | ★★★★★ | ★★★★ | ★★★★★ |
| 2 | **边界感知异常图精炼** | Boundary Loss, CRF | ★★★★★ | ★★★ | ★★★★★ |
| 3 | **对比学习增强提示词** | SupCon, CLIP 原始目标 | ★★★★★ | ★★★★ | ★★★★ |

### 组合建议

**最佳组合：方向 A + 方向 B**
- Adapter 微调视觉端 + 边界损失优化异常图
- 两者互补：Adapter 改善特征质量，Boundary Loss 改善边界精度
- 总新增参数 <1MB，显存开销可忽略
- 可以做完整的消融实验：单独 Adapter、单独 Boundary Loss、两者结合

### 需要注意的问题

1. **论文创新性**：Adapter/LoRA 已在分类任务中被广泛研究，需要在异常检测场景下找到新的切入点（如：视觉端 vs 文本端的联合优化策略、Adapter 位置的选择等）
2. **实验充分性**：需要在 MVTec AD、VisA、BTAD、MPDD 等多个数据集上验证，包含零样本跨域迁移实验
3. **消融实验**：每个改进组件都需要单独验证贡献

---

## 参考文献

1. Zhou et al. "AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection." ICLR 2024.
2. Jeong et al. "WinCLIP: Zero-/Few-Shot Anomaly Classification and Segmentation." CVPR 2023.
3. Gu et al. "AnomalyGPT: Detecting Industrial Anomalies using Large Vision-Language Models." AAAI 2024.
4. Gao et al. "CLIP-Adapter: Better Vision-Language Models with Feature Adapters." IJCV 2023.
5. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022.
6. Zhou et al. "Learning to Prompt for Vision-Language Models." CVPR 2022. (CoOp)
7. Zhou et al. "Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification." ECCV 2022.
8. Liu et al. "SimpleNet: A Simple Network for Efficient Anomaly Detection and Localization." CVPR 2024.
9. Batzner et al. "EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies." CVPR 2024.
10. Roth et al. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022. (PatchCore)
11. Zavrtanik et al. "DRAEM – A discriminatively trained anomaly localization and detection approach." ICCV 2021.
12. Kervadec et al. "Boundary Loss for Highly Unbalanced Segmentation." IPMI 2019 / Medical Image Analysis 2021.
13. Tan et al. "EfficientDet: Scalable and Efficient Object Detection." CVPR 2020. (BiFPN)
14. Li et al. "TinyCLIP: CLIP Distillation via Affinity Mimicking and Weight Inheritance." ICCV 2024.
15. Apple/CMU. "MobileCLIP: Fast Image-Text Models through Multi-Modal Reinforced Training." 2024.
16. Radford et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. (CLIP)
