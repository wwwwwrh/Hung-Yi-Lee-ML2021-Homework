# HW02 — Phoneme Classification with PyTorch

> 李宏毅《Machine Learning 2021》课程作业 HW2-1：基于 TIMIT 声学特征实现逐帧音素分类，并通过对照实验研究优化、正则化、模型容量和类别不平衡对分类性能的影响。
>
> **Best Kaggle Accuracy：Public 74.512% / Private 74.441%**（最终 30-epoch 版本，已重新提交验证）。

本项目基于[课程官方 Starter Notebook](https://github.com/ga642381/ML2021-Spring/blob/main/HW02/HW02-1.ipynb)进行实现与修改。除了完成训练和预测，也记录了从原始 Baseline 到最终方案的改进过程，以及未取得正向收益的实验及其分析。

## 1. 任务说明（Task Overview）

### 1.1 任务与数据

HW2-1 是一个 **39 类逐帧音素分类（Framewise Phoneme Classification）**任务：根据语音帧的 MFCC 特征预测其音素类别，最终使用 Kaggle Accuracy 评估预测结果。

每帧有 39 维声学特征。由于单帧信息有限，数据预先拼接目标帧**前后各 5 帧**，形成 11 帧的上下文窗口：

$$
X\in\mathbb{R}^{11\times39}
\quad\xrightarrow{\mathrm{flatten}}\quad
\mathbf{x}\in\mathbb{R}^{429}.
$$

模型输出 39 个类别的 logits：

$$
f_\theta(\mathbf{x})\in\mathbb{R}^{39},\qquad
\hat y=\arg\max_{c\in\{0,\dots,38\}}f_\theta(\mathbf{x})_c.
$$

| 文件 | 作用 | 当前 Notebook 读取的规模 |
|---|---|---:|
| `timit_11/train_11.npy` | 训练特征 | `(1,229,932, 429)` |
| `timit_11/train_label_11.npy` | 训练标签 | 1,229,932 个类别标签 |
| `timit_11/test_11.npy` | 测试特征 | `(451,552, 429)` |

最终预测文件包含 `Id` 和 `Class` 两列，每行对应一个测试帧的预测类别。**本项目不使用额外标注数据或预训练模型。**

### 1.2 方法思路

从课程提供的 Sigmoid MLP Baseline 出发，按以下顺序进行实验：

1. **改善优化过程**：使用 ReLU 替换 Sigmoid，比较不同学习率。
2. **提高训练稳定性和泛化能力**：加入 Batch Normalization、Dropout、StepLR，并测试 Weight Decay。
3. **研究模型容量**：比较不同宽度、深度及正则化强度的 MLP。
4. **分析验证可靠性**：采用不同连续区间划分进行模型选择，并在 `model.eval()` 下检查训练/验证差距。
5. **分析长尾类别**：统计 39 类样本数，测试少数类 Oversampling 与 per-class / macro accuracy。

最终以独立 Kaggle 测试成绩为主要结果；验证集仅用于选择模型与诊断实验现象。

## 2. 模型与训练实现（Model and Training）

### 2.1 最终网络

采用三层隐藏层的多层感知机（MLP）：

```text
Input (429)
  ↓
Linear(429, 1024)  → BatchNorm1d → ReLU → Dropout(0.3)
  ↓
Linear(1024, 1024) → BatchNorm1d → ReLU → Dropout(0.3)
  ↓
Linear(1024, 512)  → BatchNorm1d → ReLU → Dropout(0.3)
  ↓
Linear(512, 39)
  ↓
39-class logits → argmax
```

网络共有 **2,039,847 个可训练参数**。最终输出层不使用 Sigmoid 或 Softmax，因为 `nn.CrossEntropyLoss` 可直接接收 logits。

### 2.2 最终超参数



| 项目 | 设置 |
|---|---|
| Framework | PyTorch |
| Model | MLP：429 → 1024 → 1024 → 512 → 39 |
| Activation | ReLU |
| Normalization | 每个隐藏层使用 `BatchNorm1d` |
| Dropout | `p=0.3`，作用于三个隐藏层 |
| Loss | `nn.CrossEntropyLoss()` |
| Optimizer | `torch.optim.Adam` |
| Initial Learning Rate | `1e-4` |
| Scheduler | `StepLR(step_size=15, gamma=0.3)` |
| Batch Size | 64 |
| Epochs | **30** |
| Random Seed | 0 |
| Training Data | 全部 **1,229,932** 个有标签样本 |
| Validation Set | 最终全量训练不保留验证集 |
| Oversampling / Weight Decay | 最终模型均不使用 |
| Checkpoint | `model_hw2_best_fulltrain.ckpt`（最后一个 epoch 保存） |
| Prediction | `prediction_hw2_best.csv` |

实际训练的学习率为：第 1–15 epoch 使用 `1e-4`，第 16–30 epoch 使用 `3e-5`。最终训练不通过验证集挑选 checkpoint，而是使用此前实验选择的配置和预定训练轮次，训练完成后保存权重。

### 2.3 为什么修改这些组件？

- **ReLU**：相比原始 Sigmoid，减轻饱和激活导致的梯度过小问题，早期实验收敛更快、验证准确率更高。
- **BatchNorm**：对隐藏特征进行批归一化和可学习的仿射变换，改善优化过程；在早期对照中，验证准确率由约 71.70% 提升到 72.17%。
- **Dropout**：降低隐藏单元间的过度依赖。实验发现适当 Dropout 对泛化非常重要，但更大概率不一定更好。
- **Adam**：使用梯度一阶、二阶矩估计进行自适应更新；实验中初始学习率 `1e-4` 表现较稳定。
- **StepLR**：训练后期降低更新步长，有助于细化优化。最终版本使用 `step_size=15`，与早期划分验证集时的调度配置区分开来。
- **CrossEntropyLoss**：对应单标签 39 类分类任务，计算模型 logits 与真实标签之间的分类损失。

### 2.4 代码实现与修改范围

本项目从官方 Notebook 出发，主要进行了以下工作：

| 模块 | 实际实现或修改 |
|---|---|
| Dataset / DataLoader | 通过 `TIMITDataset` 封装 `.npy` 特征与标签；使用 `DataLoader` 按 batch 训练 |
| 数据类型兼容 | 将标签转换为 `np.int64` / `torch.LongTensor` |
| 模型结构 | 调整 MLP 隐藏层维度，引入 ReLU、BatchNorm、Dropout |
| 训练策略 | 比较 Adam、SGD with Momentum、学习率配置；增加 StepLR |
| 模型保存 | 在验证阶段保存最佳模型；全量训练阶段保存最后一轮权重 |
| 推理 | 使用 `model.eval()`、`torch.no_grad()`，输出 Kaggle 提交 CSV |
| 实验诊断 | 增加 eval-mode Train/Validation 对照、类别分布统计、Oversampling、per-class / macro accuracy |

**归属说明**：数据处理与基础训练框架来自课程 Starter Notebook；以上修改与实验在该框架基础上完成，不将课程模板声称为从零原创实现。

## 3. 实验结果（Results）

### 3.1 Kaggle 最终成绩

最终使用完整训练集训练 30 epochs，已重新验证与此前最佳成绩一致：

| Metric | Accuracy |
|---|---:|
| **Kaggle Public** | **0.74512（74.512%）** |
| **Kaggle Private** | **0.74441（74.441%）** |

Public 与 Private 之间的差距较小，说明模型在这两个测试子集上的表现接近。**尚未达到课程 Strong Baseline**，因此后续仍有优化空间。

最终全量训练的第 30 epoch 日志为：

| Training Metric | Value |
|---|---:|
| Training Accuracy | 0.750588（75.0588%） |
| Training Loss | 0.736659 |

这里的 Training Accuracy 在 `model.train()` 下统计，包含开启 Dropout 时的随机失活；不能直接拿它与 `model.eval()` 下的测试准确率判断泛化差距。最终全量训练没有单独 Validation Loss，因此不展示或推断不存在的最终验证曲线。

### 3.2 Split A：基础模型改进

早期实验使用**前 80% 训练、后 20% 验证**的连续区间划分（Split A）。

| 实验配置 | Best Validation Accuracy |
|---|---:|
| 官方 Sigmoid Baseline | 70.44% |
| Sigmoid → ReLU | 71.70% |
| 加入 BatchNorm | 72.17% |
| BatchNorm + Dropout 0.2 | 74.19% |
| 1024 → 512 → 256，Dropout 0.2 | 74.76% |
| 1024 → 1024 → 512，Dropout 0.2 | 75.06% |
| 上述宽网络加入 Weight Decay `1e-4` | 75.15% |
| **1024 → 1024 → 512，Dropout 0.3** | **75.51%** |

整体来看，ReLU、BatchNorm、Dropout 与适当扩大网络宽度带来了连续收益。Weight Decay 在所测试配置中的增益较小；不能据此认为其对所有网络都无效。

### 3.3 Split B：验证稳健性与进一步实验

后续为降低重复使用同一验证区间的模型选择偏差，使用**中间 20% 验证，其余 80% 训练**的 Split B。(此处不同验证划分的准确率不可直接作绝对性能比较。)

| 实验配置 | Best Validation Accuracy |
|---|---:|
| 1024 → 1024 → 512，Dropout 0.3 | 73.6794% |
| Adam → SGD（SGD lr=`1e-3`） | 73.5839% |
| 少数类 Oversampling 到 10,000 | 73.2733% |
| 2048 → 1024 → 512，Dropout 0.3 | 73.5570% |
| **2048 → 1024 → 512，Dropout 0.4** | **73.7225%** |
| 约 12M 参数深层 MLP，Dropout 0.5，50 epochs | 73.2887% |

这组实验显示：简单增加参数量不一定提高验证准确率；网络容量、正则化强度与训练调度需要共同设计。

## 4. 实验分析（Experimental Analysis）

### 4.1 过拟合与正则化

在 Split B 上，较宽的 `429 → 2048 → 1024 → 512 → 39` 网络使用 Dropout 0.3 时，后期 Training Loss 持续下降，但 Validation Loss 从约 `0.811` 回升至 `0.829`，表现出过拟合迹象。

将同一网络的 Dropout 提高到 0.4 后，最佳 Validation Accuracy 从 **73.5570%** 提升至 **73.7225%**，且后期 Validation Loss 更稳定。说明增加模型容量时，适当增强正则化可能改善泛化；但正则化并非越强越好。

需要注意，训练循环中的 Training Accuracy / Loss 在 Dropout 开启的 `train()` 模式下计算，而验证指标在 `eval()` 模式下计算。为了避免不公平比较，我们另外实现了统一 `eval()` 模式的 Train-Eval / Validation 诊断。

### 4.2 Adam → SGD 切换及 Scheduler 问题

我们曾尝试在训练中途从 Adam 切换到带 Momentum 的 SGD。首次将 SGD 学习率设为 `3e-5`，训练损失下降明显减缓；提高至 `1e-3` 后恢复正常优化，但最佳验证准确率 **73.5839%**，仍略低于纯 Adam 方案的 **73.6794%**。

调试过程中还发现一个 PyTorch 实现问题：重新赋值为新的 Optimizer 后，之前创建的 StepLR **仍绑定旧的 Optimizer 实例**，因此调度没有作用于实际训练的参数更新。检查绑定关系并重新安排优化器与调度逻辑后解决了该问题。

由此认识到：不同优化器所适合的学习率范围不同，而且实验中的训练状态与 Scheduler 对象必须正确管理。该方向没有展示出稳定收益，因此最终方案保留 Adam + StepLR。

### 4.3 深层大模型与训练策略

还尝试了约 1200 万参数的深层 MLP：

```text
429 → 2048 → 2048 → 2048 → 1024 → 512 → 128 → 39
```

在 Dropout 0.5 下，将 StepLR 衰减点由 20 延后到 30，并把训练延长至 50 epochs 后，最佳验证准确率由约 **72.96%** 提升至 **73.29%**，但仍未超过较小的 MLP。

对应 50-epoch checkpoint 的 eval-mode 诊断：

- Train-Eval Accuracy：**80.8065%**
- Validation Accuracy：**73.2887%**
- Generalization Gap：**7.518 个百分点**

实验说明更大的容量并不会自动转化为更好的泛化，深度、Dropout、学习率日程及训练预算都可能影响最终结果。

### 4.4 类别不平衡与 Oversampling

Split B 训练集中，类别样本数差异明显：

- 样本最多的 **Class 38：142,385** 个；
- 样本最少的 **Class 6：3,096** 个；
- 数量比约为 **45.99:1**。

为验证类别不平衡对分类的影响，对样本少于 10,000 的 **15 个类别**做随机过采样，将这些类别补到 10,000 个训练实例，总共新增 **46,789 个重复样本索引**。过采样仅作用于训练集，验证集保持原始分布。

| 指标（Split B） | 无 Oversampling | Oversampling 至 10k | 变化 |
|---|---:|---:|---:|
| **Overall Accuracy** | **73.6794%** | 73.2733% | −0.4061 pp |
| **Macro Accuracy** | 67.6333% | **68.7418%** | +1.1084 pp |

部分少数类的验证准确率显著改善：

| Class | Before | After | Improvement |
|---|---:|---:|---:|
| 6 | 16.88% | 22.72% | +5.84 pp |
| 20 | 63.51% | 69.97% | +6.46 pp |
| 24 | 59.49% | 77.32% | +17.83 pp |
| 34 | 24.94% | 37.49% | +12.55 pp |

这说明简单过采样能够改善部分长尾类别的表现，但也可能使多数类别的准确率下降。当评价指标是所有帧等权的 Overall Accuracy 时，少数类收益未必足以抵消多数类损失。**因此最终 Kaggle 模型不使用 Oversampling。**

## 5. 后续优化方向（Future Work）

1. **更精细的长尾学习**：尝试较温和的 WeightedRandomSampler、Class-weighted CrossEntropy、Focal Loss 或 logit adjustment，并共同报告 Overall Accuracy、Macro Accuracy 与 Per-class Recall，避免只针对少数类优化却损害整体指标。
2. **显式建模时序特征**：当前 MLP 将 11 帧展平。可以考虑以 `(11, 39)` 形式使用 1D CNN、轻量 RNN 等模型，学习短时声学特征的时间结构；或在合理的 utterance 边界内进行预测后处理。
3. **更严格的验证切分**：相邻帧的 11 帧上下文窗口存在高度重叠。若能获得 utterance / speaker 分组信息，应优先尝试按语句或说话人分组验证，并用多个随机种子报告波动范围，降低划分方式带来的偏差。
4. **系统化实验管理**：将模型结构、数据划分、学习率、Dropout、Scheduler、随机种子等记录在统一配置中，自动导出逐 epoch 指标和 checkpoint，方便复现实验及开展消融分析。

## 6. 运行方法（Reproduction）

本项目的主要入口为 **`HW02_solution.ipynb`**。使用 Python、PyTorch、NumPy 和 Jupyter Notebook；如果希望使用 CUDA，需要安装与本机环境匹配的 PyTorch 版本。

1. 从课程作业提供的数据来源获取 `train_11.npy`、`train_label_11.npy`、`test_11.npy`，按下列目录放置：

   ```text
   .
   ├── README.md
   ├── HW02_solution.ipynb
   └── timit_11/
       ├── train_11.npy
       ├── train_label_11.npy
       └── test_11.npy
   ```

2. 安装相关环境（例如 `pip install torch numpy notebook`）；GPU 版本的 PyTorch 请按官方安装说明选择。
3. 用 Jupyter 或 VS Code 打开 Notebook，从头执行各单元；确保工作目录为 Notebook 所在目录。
4. 训练结束后生成 `model_hw2_best_fulltrain.ckpt`；运行测试单元后生成 `prediction_hw2_best.csv`，可提交到 Kaggle 进行评估。

训练数据、模型权重和生成的预测 CSV 不必随仓库一同提交。Notebook 是课程示例基础上的修改版本，实际分数可能受软件版本、设备和随机数实现影响。

## 7. 总结（Conclusion）

本项目以一个简单的 Sigmoid MLP Baseline 为起点，通过 ReLU、BatchNorm、Dropout、网络宽度调整与学习率调度改进音素分类效果，最终取得 **Kaggle Public 74.512% / Private 74.441%**。

除了最终指标，实验还涉及优化器学习率错误、Scheduler 对象绑定、过拟合诊断、深层网络训练策略和长尾类别采样等问题。实践表明，**优化过程、模型容量、正则化和数据分布之间存在相互作用；任何单项改动的收益都应通过可比较的验证实验确认**。

## 参考资料（References）

- Hung-yi Lee, [Machine Learning 2021 — HW02 Starter Notebook](https://github.com/ga642381/ML2021-Spring/blob/main/HW02/HW02-1.ipynb).
- Hung-yi Lee, [HW02 Assignment Slides](https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/hw/HW02/HW02.pdf).
- [HW02 Lecture Video (Chinese)](https://youtu.be/PdjXnQbu2zo) / [English](https://youtu.be/ESRr-VCykBs).

> 本仓库用于课程学习与实验复现。课程框架及任务资料归原作者所有；本文所列实验分析基于实际运行记录。
