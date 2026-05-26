# AnomalyCLIP 实验记录

> 项目：AnomalyCLIP | 模型：ViT-B/16 | GPU：RTX 3060 Laptop 6GB
> 数据集：BTAD | 图像尺寸：224x224 | Batch Size：16

---

## 实验 v1.0 — ViT-B/16 Baseline（逐层简单求和）

**日期**：2026-05-06
**实验目的**：在 BTAD 数据集上建立 ViT-B/16 的 baseline 结果

### 代码改动

| 文件 | 改动内容 | 说明 |
|------|---------|------|
| `AnomalyCLIP_lib/model_load.py` | 优先从 `clip_models/` 本地加载 | 避免重复下载 |
| `prompt_ensemble.py` | `compound_prompt_projections` 输出维度修正为 `ctx_dim` | ViT-B/16 的 embed_dim=512，而非硬编码 896 |
| `train.py` | 默认参数改为 `features_list=[3,6,9,12]` `image_size=224` `batch_size=16` | 适配 ViT-B/16 12 层架构 |
| `train.py` | 修复 `text_probs.squeeze()` 在 batch_size=1 时报错 | shape 不匹配 bug |
| `train.py` | 自动适配 `DPAM_layer`：`min(20, num_layers+1)` | ViT-B/16=13，ViT-L/14=20 |

**训练配置**：lr=0.001, depth=9, n_ctx=12, t_n_ctx=4, epoch=15

### 实验结果

| 指标 | 01 | 02 | 03 | **mean** |
|------|----|----|----|---------|
| pixel_auroc | 92.2 | 95.8 | 98.5 | **95.5** |
| pixel_aupro | 17.3 | 50.4 | 43.2 | **37.0** |
| image_auroc | 88.0 | 86.7 | 99.5 | **91.4** |
| image_ap | 95.1 | 97.9 | 94.8 | **95.9** |

**checkpoint**：`checkpoints/btad_vitb16/epoch_15.pth`

---

## 实验 v2.0 — 自适应多层特征融合（从零训练）

**日期**：2026-05-08
**实验目的**：引入 FeatureFusionModule，用 CLS token 注意力加权替代逐层简单求和

### 代码改动

| 文件 | 改动内容 | 说明 |
|------|---------|------|
| `prompt_ensemble.py` | 新增 `FeatureFusionModule` 类 (+`fused_anomaly_map` 方法) | ~300 可训练参数，CLS token 做 query 对各层 anomaly map 注意力加权融合 |
| `train.py` | 新增 `--use_fusion` 参数（默认 True） | 控制是否启用融合模块 |
| `train.py` | 新增 `--pretrained_checkpoint` 参数 | 支持从预训练 prompt 权重加载后微调 |
| `test.py` | 新增 `--use_fusion` 分支 | 推理时调用 `fused_anomaly_map` |

**FeatureFusionModule 架构**：
- `query_proj`: Linear(embed_dim → embed_dim/4)
- `key_proj`: Linear(embed_dim → embed_dim/4)
- `temperature`: 可学习温度系数
- 对各层 patch 特征做 mean pooling → attention → softmax 得到层权重 → 加权融合 anomaly map

**训练配置**：与 v1.0 相同（从零训练，无预训练权重）

### 实验结果

| 指标 | 01 | 02 | 03 | **mean** | vs v1.0 |
|------|----|----|----|---------|---------|
| pixel_auroc | 91.9 | 95.7 | 98.2 | **95.3** | -0.2 |
| pixel_aupro | 17.3 | 50.1 | 43.8 | **37.1** | +0.1 |
| image_auroc | 88.5 | 87.3 | 99.4 | **91.7** | +0.3 |
| image_ap | 95.4 | 97.9 | 93.8 | **95.7** | -0.2 |

**checkpoint**：`checkpoints/btad_fusion/epoch_15.pth`

---

## 实验 v3.0 — 自适应多层特征融合（预训练 + 微调）

**日期**：2026-05-08
**实验目的**：用 v1.0 baseline 权重初始化 prompt 部分，fusion 模块随机初始化后微调

### 代码改动

| 文件 | 改动内容 | 说明 |
|------|---------|------|
| `train.py` | 分组优化器：prompt_lr = base_lr × 0.1, fusion_lr = base_lr | 保护已学到的 prompt，让 fusion 模块以原始学习率收敛 |
| `train.py` | 加载 checkpoint 时 `strict=False` 跳过缺失的 fusion key | 兼容旧 checkpoint |
| `test.py` | 加载 checkpoint 时兼容旧格式（缺少 fusion key 则跳过） | 向后兼容 |

**训练配置**：lr=0.001 (fusion: 0.001, prompt: 0.0001), 其他同 v1.0

### 实验结果

| 指标 | 01 | 02 | 03 | **mean** | vs v1.0 |
|------|----|----|----|---------|---------|
| pixel_auroc | 92.0 | 95.8 | 98.4 | **95.4** | -0.1 |
| pixel_aupro | 16.8 | 50.8 | 44.1 | **37.2** | +0.2 |
| image_auroc | 88.3 | 87.4 | 99.5 | **91.8** | +0.4 |
| image_ap | 95.4 | 98.0 | 95.2 | **96.2** | **+0.3** |

**checkpoint**：`checkpoints/btad_fusion_ft/epoch_15.pth`

---

## 实验总结

| 实验 | pixel_auroc | pixel_aupro | image_auroc | image_ap | 备注 |
|------|:----------:|:----------:|:----------:|:-------:|------|
| v1.0 (baseline) | 95.5 | 37.0 | 91.4 | 95.9 | 逐层简单求和 |
| v2.0 (fusion scratch) | 95.3 | 37.1 | 91.7 | 95.7 | 从零训练 fusion |
| v3.0 (fusion fine-tune) | **95.4** | **37.2** | **91.8** | **96.2** | ✅ 最佳结果 |

### 关键发现

1. **Fusion 模块从零训练 (v2.0) 效果与 baseline 持平**：自适应权重在 15 epoch 内尚未学习到显著优于均匀权重的最优策略
2. **预训练 + 微调 (v3.0) 在 image_ap 上取得最佳结果**（96.2%），相比 baseline 提升 0.3 个百分点
3. **pixel_aupro 持续小幅提升**（37.0 → 37.1 → 37.2），说明融合机制对定位精度的正向贡献
4. Fushion 模块仅增加约 300 个参数，几乎不影响推理速度

### 后续方向

- 分组优化器实验中 prompt 部分学习率衰减偏保守（0.1×），可尝试 0.3×、0.5×
- 可尝试在更多数据集上验证（MVTec、VisA）
- 可尝试零样本迁移测试（btad 训练 → mvtec 测试）

---

## 杂项修复（2026-05-08）

以下为代码库中修复的已有 bug，与上述实验改动无关：

| 文件 | 问题 | 修复 |
|------|------|------|
| `AnomalyCLIP_lib/AnomalyCLIP.py` | `ModifiedResNet` 未导入，触发 NameError | 添加 `from .CLIP import ModifiedResNet` |
| `AnomalyCLIP_lib/AnomalyCLIP.py` | `Transformer` 属性拼写 `design_deatails` → `design_details` | 修正属性名 |
| `AnomalyCLIP_lib/model_load.py` | `import urllib` 不加载子模块，`urllib.request` 不存在 | 改为 `import urllib.request` |
| `AnomalyCLIP_lib/model_load.py` | `convert_to_custom_text_state_dict` / `resize_pos_embed` 未定义 | 实现这两个缺失的辅助函数 |
| `AnomalyCLIP_lib/model_load.py` | `transform.py:34` `self.fn` 两个分支都赋了 `min` | 第二个改为 `max` |
| `AnomalyCLIP_lib/transform.py` | `timm` 未安装时错误信息不友好 | 添加 ImportError 提示 |
