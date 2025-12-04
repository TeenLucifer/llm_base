# dsa_reproduce DeepSeek Sparse Attention (DSA) 复现

本项目基于 Qwen2.5-0.5B 语言模型复现 DSA，模型层面修改 + 训练 + 效果测评。

## 🎯 项目概述
复现 DSA（DeepSeek Sparse Attention）：以 Qwen2.5-0.5B 为基座，使用 deepctrl-sft-data 数据集微调，对比引入 DSA 结构后的效果。

## 🔍 算法原理
参考 DeepSeek 技术报告中的示意图，DSA 包括轻量打分器（lightning indexer）和 top-k 选择器（top-k selector）两部分。

DSA 的整体流程是先用 lightning indexer 对 token 进行一次打分，仅选择重要性得分 top-k 的 token 进行 attention 计算，这样就实现了注意力计算的稀疏 Sparse。

e.g. ：输入模型的上下文长度为 128K，仅选择 top-2048 的 token 进行 attention 计算，实现了推理时的极致效率，推理成本从 128K token -> 2048 token。
<p align="center">
    <img src="./docs/dsa_framework.png">
</p>

设计上 DSA 属于一个网络结构，线性映射函数 w 加上激活函数 ReLU。
<p align="center">
    <img src="./docs/dsa_form.png">
</p>

### 核心公式在代码中的实现
从公示的描述来看，条件概率是指是输入序列x，输出序列为yw或yl的概率，因此实现上用的也是**序列似然概率**，这一点与PPO/GRPO等方法有差异。

代码中的概率实现都是对数似然概率，因此除法都转换为了减法，下面的loss计算对应了DPO损失函数的数学公式：
```python
def dpo_loss(ref_probs, probs, beta):
    def split_probs(probs):
        # label中包含chosen和rejected, 因此生成部分的概率手动拆成chosen序列和rejected序列
        len_chosen = int(len(probs) // 2)
        chosen_data = probs[:len_chosen]
        reject_data = probs[len_chosen:]
        return torch.cat(chosen_data), torch.cat(reject_data)

    # 分别计算ref_model和policy_model, chosen和rejected序列的对数似然概率
    ref_chosen_probs, ref_reject_probs = split_probs(ref_probs)
    chosen_probs, reject_probs = split_probs(probs)

    # Bradley-Terry模型的核心公式, 因为都是对数似然概率, 除法转换成减法
    # (chosen_probs - ref_chosen_probs) - (rejected_probs - ref_reject_probs)
    pi_logratios = chosen_probs - reject_probs
    ref_logratios = ref_chosen_probs - ref_reject_probs
    logits = pi_logratios - ref_logratios
    loss = -F.logsigmoid(beta*logits)
    return loss.mean()
```

## 📚 数据集
数据集太多了，挑出10000条大于1024token的就可以了，不用过完整个数据集，一千多万条太慢了

数据集选用M-A-P团队的COIG-P，是一个百万级中文人工智能偏好训练数据集，数据列如下所示
<p align="center">
    <img src="./docs/COIG-P_dataset.png">
</p>
（项目中实际用的是上述偏好数据集中的一个子集做训练，并进行了一道映射）

## 📊 效果展示

### 运行环境
- **policy模型训练**: 2小时（1 × AutoDL vGPU-32GB）

### 训练效果
<p align="center">
    <img src="./docs/dpo-loss.png" width="60%">
</p>
<p align="center">
    <em>Figure 1: DPO 训练过程中的 loss 曲线</em>
</p>
loss值随着训练的进行持续在掉，说明模型持续学习到人类偏好。（为了节省时间，没有等到训练效果稳定就停了）

## 🚀 项目部署运行

### 模型和数据集下载

```bash
# 模型或数据有网络问题可以在modelscope或者hf镜像站下载

# 下载Qwen2.5-0.5B模型
git clone https://huggingface.co/Qwen/Qwen2.5-0.5B

# 偏好数据
git clone https://huggingface.co/datasets/m-a-p/COIG-P
```

### 依赖安装

```bash
pip install uv
uv sync
```

### 训练步骤

```bash
# train policy model
python dpo-train.py
accelerate launch train.py # 兼容accelerate分布式训练
```

## 📖 参考资料

1. 本项目在b站up主[偷星九月333](https://github.com/wyf3/llm_related)的基础上二开，补充了测评对比

2. [Direct Preference Optimization](https://arxiv.org/pdf/2305.18290)，论文中提出的思想非常巧妙地消去了reward模型，[训练过ppo算法](https://github.com/TeenLucifer/ppo_reproduce)对比可以明显感受到节省了训练reward模型和critic模型的资源，而且仅训练一个模型稳定性也大幅提升。

## 🤝 贡献与交流

欢迎提交Issue和Pull Request来改进项目。这是一个探索性的学习项目，旨在分享多模态模型训练和部署的经验。

## 📄 许可证

本项目采用开源许可证，详见LICENSE文件。

---

**注意**: 这是一个以探索和学习为目标的项目，代码实现可能存在不足之处。如果您发现任何问题，欢迎提出Issue或PR。感谢您的指正和交流！
