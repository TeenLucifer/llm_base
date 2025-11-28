# 大模型相关算法及项目复现 🚀
面向模型训练、RAG、Agent 实战项目。

## 模型训练系列 🧭

覆盖 RLHF 与 VLM 全流程实践，dpo/ppo/grpo/dapo/gspo 等热门算法；手搓实现 + 框架化训练双轨并行，从原理到落地快速闭环。

1. [**grpo_reproduce**](https://github.com/TeenLucifer/llm_base/tree/master/grpo_reproduce)  
   - 复现 [grpo](https://arxiv.org/pdf/2402.03300) 与 [gspo](https://arxiv.org/pdf/2507.18071) 并对比  
   - 完全手搓，无训练框架依赖，便于深入理解算法细节  
   - 涵盖 off-policy 采样、训练分离，Deepspeed 分布式、思维链训练、LoRA 微调、显存估算等实践经验

2. [**dapo_reproduce**](https://github.com/TeenLucifer/llm_base/tree/master/dapo_reproduce)  
   - 复现 [dapo](https://arxiv.org/pdf/2503.14476) 全流程  
   - 延续手搓思路，聚焦熵坍缩、组内相对优势为 0、长序列 token loss 稀释等工程痛点的优化  
   - 可与 grpo/gspo 对比，体会 grpo 的问题及后续改进思路

3. [**ppo_reproduce**](https://github.com/TeenLucifer/llm_base/tree/master/ppo_reproduce)  
   - 复现 [ppo](https://arxiv.org/pdf/1707.06347)，包含 reward model 与 policy 训练  
   - 基于 [trl](https://hugging-face.cn/docs/trl/index) 框架，学习如何快速搭建 RM + PPO 流程  
   - 作为 RLHF “鼻祖”，实战感受 reward + actor + critic 带来的流程复杂度与资源成本

4. [**dpo_reproduce**](https://github.com/TeenLucifer/llm_base/tree/master/dpo_reproduce)  
   - 复现 [dpo](https://arxiv.org/pdf/2305.18290) 全流程  
   - 使用 `transformers` 训练框架：基础设施（反向传播、Deepspeed/Accelerate）封装+自定义 loss，介于手搓与全封装之间  
   - 与 PPO 对比，理解为何 DPO 训练路径更轻量

5. [**vlm_reproduce**](https://github.com/TeenLucifer/llm_base/tree/master/vlm_reproduce)  
   - 基于 SigLIP + Qwen2.5-0.5B 的 VLM 训练全流程（pretrain + SFT）  
   - 重点在结构对齐与特征融合，将视觉特征注入语言模型  
   - 该项目存在巨大拓展空间：  
     - 🔍 更换新视觉/语言模型（如 smolVLM2 + Qwen3-0.6B）验证性能  
     - 🀄 扩充中文问答预训练数据，提升图文问答效果  
     - 🎯 在 pretrain+SFT 后加入 RLHF，观察质量提升  
     - 📏 构建测评体系，系统评估 VLM 能力

## RAG 系列 📚
待建设，敬请期待。

## Agent 系列 🤖
待建设，敬请期待。
