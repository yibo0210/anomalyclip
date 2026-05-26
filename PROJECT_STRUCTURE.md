# AnomalyCLIP 项目结构文档

> **论文**: AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection (ICLR 2024)
> **核心思想**: 基于 CLIP 进行 target-agnostic（与物体类别无关）的可学习 prompt 训练，实现跨数据集的零样本异常检测。

---

## 一、目录总览

```
AnomalyCLIP-main/
├── train.py                    # 训练入口
├── test.py                     # 测试/推理入口
├── run_one_example.py          # 单张图像快速推理 demo
├── plot_results.py             # 实验结果可视化绘图
├── dataset.py                  # 数据加载器：读取 meta.json，生成 Dataset
├── prompt_ensemble.py          # ★ 核心：可学习 prompt + 自适应多层特征融合
├── loss.py                     # 损失函数：FocalLoss + DiceLoss
├── metrics.py                  # 评估指标：图像级/像素级 AUROC、AUPRO、AP
├── utils.py                    # 工具函数：图像变换、归一化
├── visualization.py            # 可视化：异常热力图叠加到原图
├── logger.py                   # 日志模块：同时输出到文件和控制台
├── requirements.txt            # 项目依赖
├── experiment_log.md           # 实验记录（ViT-B/16 + BTAD）
│
├── AnomalyCLIP_lib/            # ★ 核心库：CLIP 模型改造与加载
│   ├── __init__.py             #   从 model_load 导入所有公开接口
│   ├── AnomalyCLIP.py          #   ★ AnomalyCLIP 模型定义（双路注意力 + DPAM + 可学习 token）
│   ├── CLIP.py                 #   标准 CLIP 模型（ViT + Transformer 文本编码器）
│   ├── build_model.py          #   根据 state_dict 构建模型（CLIP 或 AnomalyCLIP）
│   ├── model_load.py           #   模型加载入口：下载/本地加载 + 相似度计算
│   ├── simple_tokenizer.py     #   BPE 分词器（CLIP 原始 tokenizer）
│   ├── transform.py            #   图像预处理变换（Resize + CenterCrop + Normalize）
│   └── constants.py            #   OpenAI CLIP 的数据集均值和标准差常量
│
├── generate_dataset_json/      # 数据集 JSON 生成脚本（每种数据集一个脚本）
│   ├── btad.py / mvtec.py / visa.py / mpdd.py / …
│   └── ...（共 17 个数据集脚本）
│
├── data/                       # 数据集目录（按需存放）
├── checkpoints/                # 训练保存的模型权重
├── results/                    # 测试输出结果
├── clip_models/                # 预训练 CLIP 权重本地存放
├── assets/                     # README 图片素材
└── README.md                   # 项目说明
```

---

## 二、逐文件详细说明

### 1. 入口脚本

#### `train.py` — 训练入口
- **功能**: 训练 AnomalyCLIP 的 prompt learner（可学习 prompt + 特征融合模块）
- **流程**:
  1. 加载 CLIP 模型（ViT-B/16 或 ViT-L/14@336px），冻结其全部权重
  2. 对视觉编码器的深层 Transformer block 做 **DPAM** 改造（将原始 self-attention 替换为 v-v attention）
  3. 初始化 `AnomalyCLIP_PromptLearner`（可学习的 normal/abnormal 文本 prompt）
  4. 图像编码 → CLS token + 多层 patch tokens；文本编码 → normal/abnormal 文本特征
  5. 计算 **图像级对比 loss**（image CLS ↔ text 的交叉熵） + **像素级 anomaly map loss**（Focal + Dice）
  6. 反向传播只更新 prompt learner 参数
- **关键参数**:
  - `--model_name`: CLIP 模型选择（ViT-B/16 或 ViT-L/14@336px）
  - `--features_list [3,6,9,12]`: 提取哪些中间层的 patch feature
  - `--depth 9 / --n_ctx 12 / --t_n_ctx 4`: prompt 深度和长度设置
  - `--use_fusion`: 是否启用自适应多层特征融合（默认为 True）
  - `--pretrained_checkpoint`: 预训练 prompt 权重路径，用于 fusion 微调

#### `test.py` — 测试/推理入口
- **功能**: 用训练好的 checkpoint 在测试集上评估异常检测性能
- **流程**:
  1. 加载模型和 prompt learner，恢复 checkpoint 权重
  2. 对每张测试图像：提取 CLS token + 多层 patch features → 计算 anomaly map
  3. 对 anomaly map 进行高斯平滑（sigma=4）
  4. 统计图像级指标（AUROC, AP）和像素级指标（AUROC, AUPRO）
  5. 输出按类别统计的表格
- **关键参数**:
  - `--checkpoint_path`: 训练好的权重路径
  - `--metrics`: 可选 `image-level` / `pixel-level` / `image-pixel-level`

#### `run_one_example.py` — 单张图像推理 Demo
- **功能**: 对单张图片进行异常检测并生成可视化热力图
- **用法**: `python run_one_example.py --image_path ./assets/brain.png --checkpoint_path ./checkpoints/xxx.pth`
- **输出**: 在原图同目录下生成 `anomaly_map_xxx.png`，热力图叠加到原图

---

### 2. 核心模型模块

#### `AnomalyCLIP_lib/AnomalyCLIP.py` — ★ AnomalyCLIP 模型架构（项目灵魂）
- **包含以下关键组件**:

| 类名 | 功能 |
|------|------|
| `Attention` | **v-v self-attention**：同时计算原始 QK 注意力和 v-v 注意力（k 和 q 都被替换为 v），用于 **DPAM** 机制 |
| `ResidualAttentionBlock` | 改造后的 Transformer block：支持**双路径**（原始路径 + DPAM 路径并行），从第 6 层开始分叉 |
| `ResidualAttentionBlock_learnable_token` | 文本编码器专用 block：支持在指定层**插入可学习的 compound prompt token** |
| `Transformer` | 路由调度器：根据是否启用 DPAM、是否为文本层，调用不同的 forward 路径 |
| `VisionTransformer` | 视觉编码器：包含 `DAPM_replace()` 方法，将深层 block 的 self-attention 替换为 `Attention`（v-v attention） |
| `AnomalyCLIP` | ★ 整体模型：视觉编码器 + 文本编码器 + 投影层。提供 `encode_image()` / `encode_text()` / `encode_text_learn()` |

- **DPAM (Dual-Path Attention Mechanism)** 原理:
  - 浅层（1~5 层）：普通 self-attention，单路径
  - 深层（6~24 层）：双路径并行 — **原始路径**保持语义建模 + **DPAM 路径**用 v-v attention 放大异常区域的 token 差异性
  - `DPAM_layer` 控制从顶层往下多少层被替换

- **Compound Prompt**: 文本编码器的深层（总共 depth=9 层）逐层插入可学习的 prompt token，让文本特征更好地对齐图像异常表征

#### `AnomalyCLIP_lib/CLIP.py` — 标准 CLIP 模型
- **包含**: `ModifiedResNet`（ResNet 视觉编码器）、`VisionTransformer`（ViT 视觉编码器）、`Transformer`（标准文本编码器）、`CLIP`（完整 CLIP 模型）
- **用途**: 当不需要 DPAM 和可学习 token 时使用（本项目实际使用的是 AnomalyCLIP 版本，此文件为保留的原始实现）

#### `AnomalyCLIP_lib/build_model.py` — 模型构建工厂
- **功能**: 从预训练 state_dict 自动检测模型结构参数（层数、宽度、patch_size 等），构建 `CLIP` 或 `AnomalyCLIP` 模型并加载权重
- **关键判断**: 如果传入了 `design_details`（即需要 DPAM + 可学习 token），构建 `AnomalyCLIP`；否则构建标准 `CLIP`

#### `AnomalyCLIP_lib/model_load.py` — 模型加载 + 相似度计算
- **`load(name, device)`**: 加载预训练 CLIP 模型（优先从 `clip_models/` 本地加载，其次从 OpenAI CDN 下载）
- **`compute_similarity(image_features, text_features)`**: 计算 patch 级图像特征与文本特征之间的余弦相似度（用于生成 anomaly map）
- **`get_similarity_map(sm, shape)`**: 将一维 patch similarity 重塑为二维特征图并上采样到目标尺寸

---

### 3. 提示词学习模块

#### `prompt_ensemble.py` — ★ 可学习 Prompt + 多层特征融合
- **`AnomalyCLIP_PromptLearner`**: 核心类，包含以下可学习参数：
  - **`ctx_pos` / `ctx_neg`**: 可学习的 normal/abnormal 文本 prompt 向量（维度 `[n_cls, n_template, n_ctx, dim]`）
  - **`compound_prompts_text`**: 深层文本编码器逐层插入的可学习 token（共 depth-1 层）
  - **`feature_fusion`** (FeatureFusionModule): ★ 自适应多层特征融合模块

- **`FeatureFusionModule`**（方向一改进）:
  - 用图像 **CLS token** 作为 query，对各层 patch feature 做交叉注意力加权
  - 输出 softmax 归一化的层权重，替代原始代码中的简单相加
  - 仅增加约 **300 个可训练参数**
  - **`fused_anomaly_map()`**: 用加权融合替代简单求和生成最终 anomaly map

- **`encode_text_with_prompt_ensemble()`**: 手工设计的 prompt ensemble（训练时不使用，用于 zero-shot baseline）
  - normal prompts: `"flawless {}"`, `"perfect {}"` …
  - abnormal prompts: `"damaged {}"`, `"{} with flaw"` …
  - 每个 prompt 再套用 CLIP 的 35 个模板（`"a photo of a {}"` 等）

- **`tokenize()`**: 将文本转为 BPE token，返回 `[batch, 77]` 的 token tensor

---

### 4. 数据模块

#### `dataset.py` — 数据加载器
- **`generate_class_info(dataset_name)`**: 根据数据集名称返回类名列表和类别 ID 映射（支持 mvtec / visa / mpdd / btad / SDD / DTD / colon / ISBI 等）
- **`Dataset` 类**:
  - 读取 `data/{dataset}/meta.json` 获取所有样本路径和标签
  - `__getitem__` 返回: `img`（图像 tensor）, `img_mask`（ground truth mask）, `cls_name`（类别名）, `anomaly`（0=正常/1=异常）, `img_path`, `cls_id`

#### `generate_dataset_json/` — 数据集 JSON 生成脚本（共 17 个）
- 每个脚本对应一个数据集（`btad.py`, `mvtec.py`, `visa.py`, `mpdd.py`, `SDD.py`, `DTD.py`, `br35.py`, `brainmri.py`, `covid.py`, `clinicDB.py`, `colonDB.py`, `endoTect.py`, `head_ct.py`, `isbi.py`, `kvasir.py`, `tn3k.py`, `DAGM.py`）
- **功能**: 遍历数据集目录结构，生成 `meta.json`（包含 train/test 两个阶段的样本信息列表）
- **统一格式**: 每个样本记录 `img_path`, `mask_path`, `cls_name`, `specie_name`, `anomaly`

---

### 5. 损失函数模块

#### `loss.py`
- **`FocalLoss`**: Focal Loss 实现，gamma=2，用于处理类别不平衡（异常区域像素远少于正常区域）
- **`BinaryDiceLoss`**: Dice Loss，衡量预测 mask 与 GT mask 的重叠度
- **`smooth(arr, lambda1)`**: 平滑正则化项（计算相邻像素的差异，约束 anomaly map 的空间连续性）
- **`sparsity(arr, target, lambda2)`**: 稀疏性正则化（约束 anomaly map 整体不要过激活）

---

### 6. 评估指标模块

#### `metrics.py`
- **`image_level_metrics(results, obj, metric)`**: 图像级评估
  - `image-auroc`: 用 `sklearn.metrics.roc_auc_score` 计算
  - `image-ap`: 用 `sklearn.metrics.average_precision_score` 计算
- **`pixel_level_metrics(results, obj, metric)`**: 像素级评估
  - `pixel-auroc`: 展平所有像素后计算 ROC-AUC
  - `pixel-aupro`: **AUPRO**（Area Under Per-Region-Overlap curve），对每个异常区域分别计算检出率，再汇总
- **`cal_pro_score(masks, amaps)`**: AUPRO 的核心计算逻辑，来自 [cflow-ad](https://github.com/gudovskiy/cflow-ad)

---

### 7. 工具模块

#### `utils.py` — 工具函数
- **`normalize(pred, max_value, min_value)`**: Min-Max 归一化到 [0,1]
- **`get_transform(args)`**: 根据参数构建图像的预处理变换（Resize + CenterCrop + ToTensor + Normalize），使用 OpenAI CLIP 的均值和标准差

#### `visualization.py` — 可视化
- **`visualizer(paths, anomaly_map, img_size, save_path, cls_name)`**: 批量将 anomaly map 以 JET 热力图叠加到原图上并保存
- **`apply_ad_scoremap(image, scoremap, alpha=0.5)`**: 热力图叠加核心函数，alpha 控制透明度

#### `logger.py` — 日志
- **`get_logger(save_path)`**: 创建 logger 实例，同时输出到文件（`log.txt`）和控制台，格式化时间戳

#### `plot_results.py` — 实验结果绘图
- **功能**: 使用 matplotlib 绘制 BTAD 实验的柱状图对比（baseline vs fusion scratch vs fusion fine-tune）
- **输出**: `results/btad_comparison.png` 和 `results/btad_improvement.png`

---

### 8. 其他支撑文件

#### `AnomalyCLIP_lib/transform.py` — 图像预处理
- **`image_transform(image_size, is_train, mean, std)`**: 构建 torchvision 的 Compose 变换管道
- **`ResizeMaxSize`**: 按最长边缩放到固定尺寸（保持宽高比，短边补零填充）
- 支持 timm 数据增强（训练时），BICUBIC 插值

#### `AnomalyCLIP_lib/constants.py` — 常量
- `OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)`
- `OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)`
- 这是 OpenAI 原始 CLIP 训练时的 ImageNet 标准化参数

#### `AnomalyCLIP_lib/simple_tokenizer.py` — BPE 分词器
- 从 OpenAI CLIP 原始代码移植的 SimpleTokenizer
- 使用 Byte-Pair Encoding (BPE)，词表大小 49152
- 特殊 token: `<|startoftext|>` 和 `<|endoftext|>`
- 依赖 `bpe_simple_vocab_16e6.txt.gz` 词表文件

#### `requirements.txt` — 项目依赖
- PyTorch 2.0.0 + torchvision 0.15.1
- scikit-learn / scikit-image / scipy（评估指标）
- timm（可选，用于训练时数据增强）
- tqdm（进度条）
- tabulate（结果表格输出）

---

## 三、数据流全景图

```
数据准备阶段:
  generate_dataset_json/btad.py  →  data/btad/meta.json

训练阶段 (train.py):
  meta.json  →  dataset.py::Dataset  →  DataLoader
  ↓
  ViT Encoder (DPAM改造)  →  CLS token + 多层 patch features
  ↓                                        ↓
  FeatureFusionModule  ←── CLS token       Text Encoder (compound prompt)
  ↓                                        ↓
  融合 anomaly map                         normal/abnormal text features
  ↓                                        ↓
  FocalLoss + DiceLoss                    CrossEntropy (image-level)
  ↓
  Backward → 只更新 prompt_learner 参数

推理阶段 (test.py):
  测试图像  →  ViT Encoder  →  CLS + patch features
  ↓                              ↓
  fused_anomaly_map()  ←──  text features (prompt_learner.forward)
  ↓
  高斯平滑  →  metrics.py  →  AUROC / AUPRO / AP
```

---

## 四、关键创新点总结

| 编号 | 创新点 | 对应代码位置 | 说明 |
|------|--------|-------------|------|
| ① | **DPAM** (Dual-Path Attention Mechanism) | `AnomalyCLIP_lib/AnomalyCLIP.py:Attention` + `DAPM_replace()` | 将深层 self-attention 的 K/Q 替换为 V，使异常区域 token 的注意力分布异于正常 token |
| ② | **Object-agnostic 可学习 Prompt** | `prompt_ensemble.py:AnomalyCLIP_PromptLearner.ctx_pos/ctx_neg` | 不绑定具体物体类别名，学习通用的"正常/异常"语义表征 |
| ③ | **Compound Prompt** | `prompt_ensemble.py:compound_prompts_text` + `AnomalyCLIP_lib/AnomalyCLIP.py:ResidualAttentionBlock_learnable_token` | 文本编码器深层逐层插入可学习 token，增强跨层语义对齐 |
| ④ | **自适应多层特征融合** (FeatureFusionModule) | `prompt_ensemble.py:FeatureFusionModule` + `fused_anomaly_map()` | 用 CLS token 做 query 对各层 anomaly map 做注意力加权融合，替代简单求和，仅增加约 300 参数 |

---

## 五、扩展自定义数据集

1. 参照 `generate_dataset_json/` 中的脚本编写新数据集的 JSON 生成器
2. 在 `dataset.py::generate_class_info()` 中添加新数据集的类别信息
3. 数据集目录结构需遵循：
   ```
   data/{dataset_name}/
   ├── meta.json
   ├── {class_name}/
   │   ├── ground_truth/
   │   │   └── {defect_type}/
   │   │       └── *.png (mask)
   │   ├── test/
   │   │   ├── ok/ (normal images)
   │   │   └── {defect_type}/ (anomaly images)
   │   └── train/
   │       └── ok/ (training normal images)
   └── ...
   ```
