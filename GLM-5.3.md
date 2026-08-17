# 你负责完成代码实现以及模型设计的上层规划，输出详细的 PROMPT 与 方案。具体的代码实现我会交给 Kimi K2.7 Code

基于 pi-r2 (https://pi-r2-flow.github.io/) 的引入光流的高频 Pi0.5

受到 pi-r2 工作的启发，以 Pi0.5 JAX 作为 Backbone，用快慢通道提高它的动作生成中的视觉闭环反馈能力。

实现：与 pi-r2 相同，每次一步去噪生成动作并进行下发，每次生成动作的 VLM 在慢通道中异步作为全局视觉。光流则作为快通道实时矫正全局视觉的动作偏置。
具体实现：
1. 光流模型使用 SEA-RAFT（取最后一次 refinement 后，上采样前的 flow_8x）
2. Flow Tokenizer : 用经典的 CNN -> Flatten -> MLP 将 Flow Token化。具体结构你自行合理设计。记得 PosEmb。
3. 如何引入 Flow 到 Action Expert: 
    - 在 Action Expert 的内部中后层，将其层的输出 Hidden A 与 Flow Token 进行 Cross-Attn（每一层维护不同的 Q，K，V 映射，Hidden A 作为 Q， Flow Token 作为 K 和 V）
    - 同时每一层维护一个 Gate矩阵(从0初始化)，将其对 Cross-Attn 的输出进行门控。让模型慢慢引入 Flow。
    - 最后直接将门控输出与原 Hidden A 进行线性相加。
4. 在哪些层引入？假设AE 有 L 层，建议在 0.4L 0.65L 0.9L 的位置(近似)分别引入三次(你可以选择更合理的层数与位置)

## 训练时：不要全量微调。具体的冻结与学习部分你自己设定。

规划要求：
以上是 flowPi 的基本架构设计，请以此为主体，不要重新设计整体方案。你需要先结合当前 π0.5 JAX repository 的真实实现，对尚未明确的技术细节进行补全，并给出具体的上层实现规划。

重点补充：

Flow Tokenizer 的具体结构、维度、normalization 和 PosEmb；
Flow Cross-Attention 的具体结构以及如何嵌入原 Action Expert block；
Gate 的具体形式；
根据真实 AE 层数确定具体 injection layers；
各模块初始化、freeze / trainable 策略；
Flow、VLM 和 action generation 的 temporal alignment；
checkpoint compatibility；
需要新增/修改的代码模块及实现顺序；

原则上保持本文给定的整体架构不变。如果某个细节在真实代码结构下明显不合理，可以调整该细节并解释原因，但不要未经说明改变核心设计。

先进行 repository analysis 和 architecture planning，不要立即大规模写代码。

## 维度如何确定？我会给你一封我训练用的 lerobot v3 数据，里面是 50Hz 的高频数据，包含光流以及光流的 mask 等，具体你自行查看。另外我训练用的数据是 lerobot_v3，你对 pi05 进行适配。

## 本机没有 CPU ，所以测试只能无 GPU 进行。最终我需要训练与推理的脚本。另外 uv 环境我还没装。

# 你只负责规划！不负责实现，我告诉你的“实现”，你转告在你的规划输出中。