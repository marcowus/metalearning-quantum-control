# 方法对比报告：MAML vs Koopman MPC with PDP

本报告对比了本项目中使用的两种自适应量子控制方法：
1.  **Original MAML (Model-Agnostic Meta-Learning)**: 基于梯度的元学习，训练一个神经网络策略 $\pi_\theta$ 的初始化权重。
2.  **Koopman MPC with PDP (Pontryagin Differentiable Programming)**: 基于物理模型的元学习，训练一个线性模型预测控制 (MPC) 控制器内部的代价权重 $Q, R$。

---

## 1. 核心机制对比

| 特性 | Original MAML (Neural Network) | Koopman MPC with PDP |
| :--- | :--- | :--- |
| **策略表示** | 神经网络 (Black-box) | 模型预测控制 (Structured Optimization) |
| **物理模型** | 不显式包含 (Implicitly learned) | 显式包含 (Linearized Koopman Dynamics) |
| **适应方式** | 在线梯度下降 (Few-shot gradient steps) | 调整优化器权重 (Weight learning) 或 闭环校正 |
| **约束处理** | 软约束 (Clipping / Penalty) | **硬约束** (Explicit in QP solver) |
| **可解释性** | 低 (很难解释为何输出某个脉冲) | **高** (通过 $Q, R$ 权重理解控制意图) |
| **计算成本** | 推理快 (Forward pass)，训练慢 | 推理慢 (Solve QP)，训练极慢 (Bilevel optimization) |

---

## 2. 实验结果分析

我们运行了对比实验 (`experiments/method_comparison.py`)，主要观察以下指标：

### (1) 适应速度 (Sample Efficiency)
*   **MAML**: 能够极快地（<5 步）适应到一个不错的水平。因为它直接优化最终的 Fidelity 目标。
*   **Koopman MPC**: 如果线性模型近似得好，它**甚至不需要适应**（Zero-shot），直接就能给出很好的控制。如果模型有偏差，通过 PDP 学习 $Q, R$ 可以弥补，但这个过程类似于“系统辨识+控制整定”，通常比单纯的梯度微调要慢一些。

### (2) 鲁棒性与约束
*   **MAML**: 输出的脉冲可能会在最后时刻违反幅度限制，或者产生高频抖动（Slew rate problem），除非在 Loss 里加很重的惩罚。
*   **Koopman MPC**: **天生优势**。无论怎么学，MPC 求解器保证输出的控制脉冲严格满足 $|u| \le u_{max}$ 且平滑（如果在 MPC 里加了速率约束）。这对于**实际硬件实验**至关重要。

### (3) 最终性能 (Fidelity)
*   **MAML**: 理论上限更高。神经网络是通用的函数近似器，可以学会应对高度非线性的系统动力学。
*   **Koopman MPC**: 受限于 Koopman 线性嵌入的准确度。如果量子系统的非线性极强（例如强耦合、大幅度旋转），线性模型预测的误差会限制 MPC 的上限。

---

## 3. 结论：哪个更好？

**没有绝对的赢家，取决于应用场景。**

### 选择 **MAML** 如果：
1.  **系统非线性极强**，且很难找到好的观测函数（Observables）来线性化。
2.  **推理延迟要求极低**（例如纳秒级实时反馈），神经网络前向传播比解 QP 快几个数量级。
3.  追求**极限保真度**，不介意控制波形稍微有点“奇怪”。

### 选择 **Koopman MPC** 如果：
1.  **硬件约束严格**（例如 DAC 有严格的幅度和带宽限制）。
2.  需要**可解释性**和**安全保证**（MPC 保证不会输出疯狂的电压值）。
3.  **Sim-to-Real 差距主要在于模型参数**，而不是模型结构。通过 PDP 学习到的 $Q, R$ 往往能很好地迁移到真机上。

### 建议
在本项目（Meta-Quantum Control）的背景下，**Koopman MPC 是一个非常有价值的补充**。它提供了一条通往“安全、受限、可解释”控制的路径，是对纯黑盒 MAML 的有力增强。未来的工作可以将两者结合：**用 MAML 来初始化 Koopman MPC 的权重**，或者用神经网络来作为 Koopman 的非线性观测函数（Deep Koopman MPC）。
