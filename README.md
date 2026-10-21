# ML2021 PyTorch Learning

> 李宏毅《Machine Learning 2021》课程学习与 PyTorch 实验记录  
> A personal learning log and reproducible homework experiments based on Hung-yi Lee's ML2021 course.

本仓库记录我系统学习机器学习与深度学习的过程。目前以 **PyTorch + Jupyter Notebook** 完成课程作业，并侧重整理模型实现、训练与评估流程，以及实验中的问题和反思。

**项目定位：** 课程学习与实验复现仓库。作业均以课程给出的任务和 Starter Notebook 为基础，具体修改、参考来源和实验限制详见各作业目录的 README。

## 1. 学习目标

- 理解监督学习中的回归、分类、损失函数、前向与反向传播及梯度优化。
- 熟悉 PyTorch 的 `Dataset`、`DataLoader`、`nn.Module`、训练循环、模型保存与推理。
- 学习通过训练/验证指标、对照实验和错误分析判断模型是否真正改善。
- 随课程进度逐步学习 CNN、Self-Attention、Transformer 等模型，并持续记录实践结果。

## 2. 作业导航与进度

| 作业 | 内容 | 已完成的实验重点 | 结果 | 状态 |
| --- | --- | --- | --- | --- |
| [HW01 — Regression](HW01-Regression/README.md) | COVID-19 阳性率预测 | 15 维特征、Train-only Z-score、MLP、学习率/优化器/正则化对比 | Kaggle **Public RMSE 0.89720 / Private RMSE 0.89980** | ✅ 已完成 |
| [HW02 — Classification](HW02-Classification/README.md) | TIMIT 逐帧音素分类（39 类） | MLP、BatchNorm、Dropout、StepLR、类别不平衡实验 | Kaggle **Public Accuracy 74.512% / Private Accuracy 74.441%** | ✅ 已完成 |
| HW03 — CNN | 图像分类 | 后续学习并整理 | — | ⏳ 计划中 |

> 以上结果摘自本人作业实验记录，不同硬件、框架版本或随机实现也可能导致结果略有变化。具体实验配置、模型对比和问题分析均记录于各目录 README。

### 主要学习收获

**HW01 — [实验详情](HW01-Regression/README.md)**  
从课程回归 Baseline 出发，比较模型深度、优化器、Batch Size 和正则化，重点尝试特征选择与标准化。最终模型为 **15 → 64 → 1 MLP**（SGD + Momentum），在固定 Validation 划分下选择最佳 checkpoint。实验使我注意到：**更低的单次验证误差不一定意味着隐藏测试集表现更好**；在小数据任务中，特征设计与正确的预处理可能比增加模型规模更重要。

**HW02 — [实验详情](HW02-Classification/README.md)**  
在官方音素分类 Baseline 上，研究 ReLU、BatchNorm、Dropout、网络宽度和学习率调度的影响；最终采用 **429 → 1024 → 1024 → 512 → 39 MLP**。进一步测试了类别不平衡与过采样，发现部分少数类表现改善的同时，整体 Accuracy 可能下降；实验也帮助我理解了训练/评估模式的区别，以及优化器与 Scheduler 的绑定关系。

## 3. 仓库结构

```text
ML2021-PyTorch-Learning/
├── README.md                    # 总览、学习目标和复现说明
├── requirements.txt             # Python / PyTorch / Notebook 依赖
├── .gitignore                   # 排除数据、权重和本地环境
├── HW01-Regression/
│   ├── HW01_solution.ipynb     # HW1 最终训练和预测代码
│   ├── README.md               # 任务、方法、实验结果与分析
│   └── assets/
│       ├── learning_curve.png
│       └── validation_predictions.png
├── HW02-Classification/
│    ├── HW02_solution.ipynb     # HW2-1 最终训练和预测代码
│    └── README.md               # 任务、方法、实验结果与分析
│......
 ```

## 4. 来源与引用

- **课程：** [李宏毅 Machine Learning 2021](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.php)
- **课程作业与 Starter Notebook：** [ga642381/ML2021-Spring](https://github.com/ga642381/ML2021-Spring)


