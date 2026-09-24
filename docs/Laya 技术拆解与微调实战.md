# Laya 技术拆解与微调实战：让模型直接做判断，而不是生成一段答案

一封退款邮件进来，系统真正需要的，可能只是三个判断：该交给哪个部门处理、事情有多紧急、客户有没有流失风险。

为了得到这几个结果，先调用一个大模型，再等它逐字输出 JSON，最后检查格式、解析字段、处理异常，多少有点绕。既然最后只需要标签和概率，能不能让模型直接把这些结果算出来？

**Laya 走的就是这条路线：不生成回答，而是根据输入、问题和候选项，直接预测结构化决策的概率分布。** 原文介绍的是一个约 4.21 亿参数的非自回归决策模型，底座为双向编码器 ModernBERT-large，上面叠加专门的决策头。

![](https://i-blog.csdnimg.cn/direct/9fbe8a3b145d4d0a8aed59bba0c28821.png)


*图 1：Laya 官方标识——Decisions, not text。来源：项目 GitHub 仓库。*

> https://github.com/NandhaKishorM/laya

>官方博客：https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me
## 一、Laya 到底是什么：一个决策模型，不是一套完整 Agent 框架

先把概念拆开。

Laya 的核心是模型，外围提供加载、推理和部署接口。它本身并不等于一个能够自主规划、调用任意工具、完成复杂任务的完整智能体。更合适的位置，是作为系统中的一个判断模块：收到状态，回答定义好的问题，再由业务代码决定下一步怎么走。

这和传统文本分类有相似之处，但输入形式更灵活。

普通分类器经常把标签空间固定在输出层里，例如始终在“财务、技术、销售”三个类别中选择。Laya 则把问题、候选标签以及标签解释也放进输入，让模型对这些候选项逐一打分。

换句话说，它学习的是：

> 结合这份材料，哪个候选项更符合当前问题？

而不只是记住一个固定类别编号。原文通过问题类型、指令、候选描述与输入状态的联合编码，实现这种决策方式。

不过，**接口允许临时换问题，不代表模型对所有新领域都能准确判断**。动态候选项解决的是接口与建模方式的问题，泛化能力仍然要靠数据和评测验证。

原文还介绍了作者的技术演进：早期方案使用冻结的序列表征和独立 PPO 网络，预测销售对话的转化趋势；后来转向端到端编码器和通用决策头。至于作者与其他产品之间的“谁先提出”争论，仅凭这篇文章不足以作出独立判断，也不是理解 Laya 的必要前提。

## 二、三个决策原语：选择、评分与真假概率

Laya 接收一个 `state` 和一个或多个 `questions`。

`state` 可以是文本、JSON 对象或对话内容，`questions` 则规定需要作出哪些判断。原文将它们归纳为三种类型：

| 类型       | 输入怎样定义      | 输出的核心含义              | 常见用途           |
| -------- | ----------- | -------------------- | -------------- |
| `choice` | 候选标签及各自的解释  | 选中的标签，以及各候选项的概率      | 部门路由、意图识别、主题分类 |
| `score`  | 从低到高排列的等级描述 | 等级分布，以及等级编号的期望值      | 紧急程度、回答质量、风险等级 |
| `noul`   | 一个真假问题      | `P(true)`，即命题为真的预测概率 | 是否钓鱼、是否威胁取消服务  |

`noul` 是项目使用的类型名称，不是拼写错误，也不是让模型生成一个布尔字符串。它回答的是概率：例如 `0.82` 表示模型给“命题成立”分配了 82% 的概率质量。**这个数是否与实际发生频率一致，还要另外校准和验证。**

`score` 也需要特别说明。

假设紧急程度有三个等级：0 表示常规处理，1 表示应尽快处理，2 表示正在阻断业务。模型输出如下分布：

```text
P(0) = 0.10
P(1) = 0.30
P(2) = 0.60
```

对应的期望分数为：

$$
\operatorname{score}
=\sum_{i=0}^{K-1}i\,p_i
=0\times0.10+1\times0.30+2\times0.60
=1.50
$$

这里的 `1.50` 不是“1.5 级正确答案”，而是整个等级分布的一个摘要。真正接入业务时，应同时保留概率分布；两个期望相近的分布，可能意味着不同的不确定性。

结构化输出确实避免了自由生成带来的部分格式问题，但不能据此说模型“不会幻觉”或者“不会犯错”。

**它不会凭空写出一段解释，却仍然可能把工单分错、把攻击漏掉，或者对错误结论给出很高的概率。**

## 三、原理拆解：ModernBERT 如何变成决策模型

### 3.1 整体结构：编码器负责理解，决策头负责比较

按照原文，Laya 的主体结构可以概括为下面这条链路：

```text
输入状态 + 一个问题 + 候选项及描述
                  │
                  ▼
[CLS] 问题类型与指令 [SEP]
[MASK] 候选项 A [MASK] 候选项 B ... [SEP] 状态 [SEP]
                  │
                  ▼
ModernBERT-large：约 395M 参数
                  │
                  ▼
加入问题类型嵌入：choice / score / noul
                  │
                  ▼
两层 Transformer 决策头：约 25.2M 参数
                  │
          ┌───────┴─────────┐
          ▼                 ▼
提取各 [MASK] 位置向量    提取 [CLS] 向量与分布特征
          │                 │
          ▼                 ▼
共享 MLP 对候选项打分     Act / Escalate 决策头
          │                 │
          ▼                 ▼
温度缩放 + Softmax       自动执行 / 升级处理
          │
          ▼
标签、等级期望、概率分布
```

原文中的 ModernBERT-large 具有 28 层、1024 维隐藏状态和 16 个注意力头。两层额外 Transformer 决策头采用 1024 维表示、16 个注意力头、4096 维前馈层，再加上候选项评分器和执行决策头，总体约为 421M 参数。

这里有一处需要补充核对：

**双向注意力，不等于每一层都对整段文本做全局注意力。**

原文把 ModernBERT 描述得过于简化。ModernBERT 实际交替使用全局注意力和局部滑动窗口注意力，以兼顾跨文本交互与计算效率。

![ModernBERT 全局注意力与局部注意力交替示意](https://i-blog.csdnimg.cn/img_convert/7b07048dd42306e473c282c34a822fd7.png)

*图 2：左侧是每层全局注意力，右侧是全局与局部注意力交替。来源：Answer.AI 的 ModernBERT 官方介绍。该图说明的是底座结构，不是 Laya 完整架构。*

另一个容易混淆的地方是上下文长度。底座支持 8192 tokens，不代表某个 Laya checkpoint 的默认推理配置就会使用 8192。实际还有 `max_len` 和 `head_max_len` 等输入预算，需要检查具体模型配置。

### 3.2 `[MASK]` 在这里不是填空题，而是候选项的读取位置

假设系统要判断一封邮件由哪个部门处理，拼接后的输入可以写成：

```text
[CLS] choice question: Which team should handle this email? [SEP]

[MASK] billing: invoices, payments, refunds
[MASK] technical: bugs, outages, system errors
[MASK] other: everything else

[SEP]

{
  "subject": "Refund request",
  "body": "The customer was billed twice."
}

[SEP]
```

每个候选项前面都有一个 `[MASK]` 标记。经过编码器和决策头之后，代码取出这些标记位置的隐藏向量，再用同一个评分网络把每个向量映射成一个标量 logit。

因此，这里的 `[MASK]` 不是要求模型补出一个被遮住的词，而是一个明确的读取位置：

**让模型在这个位置汇总“当前状态与这个候选项是否匹配”的信息。**

设第 \(i\) 个候选项的表示为 \(h_i\)，共享评分器为 \(f_\theta\)，则：

$$
z_i=f_\theta(h_i)
$$

再通过温度缩放和 softmax 得到概率：

$$
p_i=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}
$$

候选项数量变化时，评分器并不需要重新增加输出神经元。它只需要对当前输入中出现的候选项分别打分，再在这一组候选项内部归一化。

当然，“能放进输入”也有成本。候选项太多、说明太长，会挤占问题与状态的 token 预算。当前输入构造代码确实会根据预算截断候选描述和状态。先检查模型到底看到了什么，往往比先调学习率更有价值。

### 3.3 “一次前向回答多个问题”，不等于状态只编码一次

原文强调，同一封邮件可以一次回答部门、紧急程度、流失风险等多个问题。这里的“一次”，主要指把多个问题组织成一个 batch，交给模型并行处理。

例如一封邮件有五个问题，内部通常是五条“状态 + 问题 + 候选项”序列，而不是先把邮件编码一次，再免费读取五个答案。

当前实现可以复用状态的分词结果，但这和共享编码后的隐藏表示是两回事。

所以，问题数量、输入长度和候选项数量增加，计算量仍然会增加。

**非自回归省掉的是逐 token 生成答案的解码过程，不是把所有计算都省掉了。**

## 四、RLCD：优化的不是一句“很有信心”，而是一整份概率分布

Laya 原文把训练方法称为 RLCD，即 Reinforcement Learning for Calibrated Decisions。

核心想法是：奖励模型给出合理的概率分布，而不只是奖励“最终猜对了”。

### 4.1 为什么只奖励“答对”，不一定能得到诚实概率

假设某类样本的真实标签分布为 \(p\)，模型按照自己的分布 \(q\) 抽样作答。如果奖励只有答对得 1、答错得 0，那么期望奖励为：

$$
\mathbb{E}[R]=\sum_k p_kq_k
$$

最大化这个式子，通常会把概率全部压到真实分布中最常见的类别上，而不是让 \(q\) 复原整个 \(p\)。

它追求的是在这套奖励下尽量多拿分，不是报告完整的不确定性。这解释了原文对朴素正确率奖励的担忧。

但原文对交叉熵的另一段描述需要纠正：

**交叉熵本身并不是“不诚实概率”的根源。**

对于真实分布 \(p\)，交叉熵可以分解为：

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)
$$

在理想条件下，最小值恰好在 \(q=p\) 时取得。实际神经网络的过度自信，还涉及有限样本、过拟合、模型设定和分布变化等因素，不能简单归结为“用了交叉熵”。对数评分本来就是严格适当评分规则的一种。

### 4.2 严格适当评分规则：报告真实分布，期望得分才最高

原文采用的理论基础是 *strictly proper scoring rules*，可译为“严格适当评分规则”。

设模型报告分布 \(q\)，实际结果 \(y\) 来自分布 \(p\)。如果一个评分函数满足：

$$\mathbb{E}_{y\sim p}[S(q,y)]\leq\mathbb{E}_{y\sim p}[S(p,y)]$$

并且只有 \(q=p\) 时等号成立，那么从期望收益看，模型最划算的选择就是报告真实分布。

原文给出的组合奖励为：

$$R(q,y)=S_{\log}(q,y)+0.5S_{\mathrm{sph}}(q,y)-\mathbf{1}_{\mathrm{score}}\operatorname{RPS}(q,y)$$

这里有三个组成部分。

**对数评分**主要惩罚“给真实结果极低概率”的情况：

$$
S_{\log}(q,y)=\sum_k y_k\log q_k
$$

**球面评分**衡量预测分布与目标分布在方向上的一致程度：

$$
S_{\mathrm{sph}}(q,y)=
\frac{\sum_k y_kq_k}
{\sqrt{\sum_k q_k^2}}
$$

**RPS，即 Ranked Probability Score**，处理的是有顺序的等级问题。

对于 0、1、2、3 四级紧急程度，把实际第 3 级预测成第 2 级，应该比预测成第 0 级受到更轻的惩罚。RPS 比较累计分布：

$$
\operatorname{RPS}(q,y)=
\frac{1}{K-1}
\sum_{i=0}^{K-2}
\left(
\sum_{k=0}^{i}q_k-\sum_{k=0}^{i}y_k
\right)^2
$$

因此，RPS 只应用于有序的 `score` 问题，不应直接套到“财务、技术、销售”这种没有自然顺序的类别上。

这里仍然需要把理论与实现分开。原文和实现都涉及数值截断，例如对数下限；再加上有限数据与不完全优化，不能把上述理论直接翻译成“训练出来的概率已经得到数学保证，放进生产就能用”。

**严格适当评分规则给出了好的优化目标，不是免评测证明。**

### 4.3 原文怎样做策略更新

原文不是只对一个标签进行采样，而是在 logits 上加入高斯噪声，探索不同的候选概率分布。

每组采样数为 8，噪声标准差从 1.0 下降到 0.3；噪声还会减去自身均值，因为给全部 logits 加上同一个常数不会改变 softmax。

对第 \(g\) 次采样，有：

$$
\tilde z^{(g)}=z+\epsilon^{(g)}
$$

$$
q^{(g)}=\operatorname{softmax}(\tilde z^{(g)})
$$

接着计算奖励，用组内平均奖励构造优势，再通过 REINFORCE 风格的策略梯度更新模型：

$$
A_g=
\frac{R_g-\overline R}
{\operatorname{std}(R)+\varepsilon}
$$

$$
\mathcal{L}_{\mathrm{policy}}=
-\frac1G\sum_g
A_g\log\pi_\theta(\tilde z^{(g)}\mid x)
$$

可以把它理解为：同一道题，模型试探几种概率分配；相对更符合目标的分配，得到更强的学习信号。

**补充核对：当前官方微调 Notebook 并不是原文所说的“零交叉熵、纯策略梯度”。**

可见代码使用了策略损失与软标签交叉熵的组合，采样数为 4，噪声标准差从 0.4 下降到 0.1，球面评分系数为 0.75。

因此，介绍原理时可以保留原文的 RLCD 思路；真正复现时，应以所固定版本的训练代码为准，不能把文章里的超参数直接当作当前微调配方。

## 五、置信度与升级处理：最容易被宣传文案带偏的地方

### 5.1 分布集中，不代表预测一定可靠

原文使用归一化熵定义置信度：

$$C_H=1-\frac{H(p)}{\log K}=
1+\frac{\sum_k p_k\log p_k}{\log K}
$$

当各个候选项概率相同时，\(C_H=0\)；当概率全部集中在一个候选项上时，\(C_H=1\)。

但这个量回答的是“概率分布有多集中”，而不是“答案有多大概率正确”。

举个计算例子，二分类分布为 `[0.9, 0.1]` 时，最大类别概率是 `0.9`，归一化熵置信度却约为 `0.531`。

两个数没有矛盾，它们衡量的不是同一件事。

当前 SDK 也体现了这个区别：`choice`、`score` 的 `confidence` 沿用熵置信度，`answer_confidence` 表示最大候选概率；`noul` 的字段语义又有所不同，其答案置信度对应 `max(P(true), P(false))`。接接口时，应检查实际版本返回值，不要把所有名为 `confidence` 的字段当成同一个统计量。

**校准的含义是：在一批预测概率相近的样本里，实际发生频率与预测概率大致一致。计算一次熵，并不会自动完成校准。**

### 5.2 Act / Escalate 头如何考虑业务代价

原文还设计了一个“自动执行还是升级处理”的决策头。

它把 `[CLS]` 表示与四个分布特征拼接：最大概率、前两名概率之差、归一化熵，以及候选数量与 255 的比值。随后通过一个小型 MLP 输出执行与升级处理的概率。

在原文的示例代价中：

* 自动执行正确，得 `+1`。
* 自动执行错误，得 `-3`。
* 升级处理，得 `-0.5`。

设自动执行正确的概率为 \(p\)，则自动执行优于升级处理的条件是：

$$
p\times1+(1-p)\times(-3)>-0.5
$$

整理得到：

$$
p>0.625
$$

这个 `0.625` 来自特定代价假设，不是万能阈值，更不是把熵置信度设成 `0.625` 就能获得相同效果。

还有一个实操细节：当前官方微调示例没有给这个执行头提供有效的独立训练目标。完成分类微调，不代表自动执行策略也已经适配了新业务。

因此，本文附带的教学脚本明确不训练执行头，部署时也不依赖它。业务升级策略应根据验证集上的错误代价、自动处理覆盖率和可接受风险另外确定。

## 六、多轮任务与训练数据：判断必须只看到当时可见的信息

原文的早期应用是销售对话转化预测。这里有个很实际的问题：

预测第 1 轮是否会转化时，绝不能把第 8 轮已经付款的消息一起编码进去。

否则模型看起来很准，其实只是偷看了答案。

原文的方法是把对话切成逐轮增长的前缀：第 \(t\) 轮的输入，只包括到这一轮为止已经发生的内容。对于最终结果 \(y\)，其递推目标写成：

$$
G_T=y
$$

$$
G_t=(1-\lambda)V(s_{t+1})+\lambda G_{t+1}
$$

在原文的终局结果设定下，\(\lambda=1\) 时，各轮直接学习最终观测结果，不再用下一轮模型预测作自举目标。

直观地说，就是学习“处在这种早期对话状态时，后面通常会发生什么”。这仍然是结果预测，并不能单凭这个训练过程推出某种回复导致了成交的因果结论。

数据切分也必须遵守相同原则。同一段对话的不同前缀，不能一部分进入训练集，另一部分进入测试集。否则即便每条输入都没有未来消息，评测仍然可能泄漏。

原文宣称，早期训练使用了真实公开数据，覆盖客服意图、事实推断、内容安全、注入攻击、质量评分和多轮对话等任务，并通过候选项顺序变化、问题改写、文本与 JSON 形式切换等方式减少模型走捷径。

但这不应该被扩展成“Laya 所有公开训练数据都没有合成成分”。

**当前微调示例使用的 `LocalLLaMA/typed-decisions` 数据卡明确包含合成流程和教师模型软标签。** 对这样的数据进行评测，很大程度上是在衡量与教师标注的一致程度，而不等于独立的人类正确率。

## 七、先跑通推理：不需要等待模型逐字生成 JSON

先安装推理包：

```bash
python -m pip install laya
```

下面沿用原文的英文邮件场景，展示三种决策类型。代码不预填任何“实测结果”，实际输出以所加载模型为准。接口形式与原文示例一致。

```python
import json
import laya

agent = laya.load("convaiinnovations/laya")

state = {
    "subject": "Charged twice on the invoice",
    "body": (
        "The customer was charged twice. They request a refund today "
        "and say they will cancel their subscription otherwise."
    ),
}

questions = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this email?",
        "criteria": {
            "billing": "Invoices, payments and refunds",
            "technical": "Bugs, outages and system errors",
            "sales": "Pricing and new contracts",
            "other": "Everything else",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": [
            "Routine handling is sufficient",
            "Needs prompt attention",
            "A critical deadline or blocking issue requires immediate action",
        ],
    },
    "churn_risk": {
        "type": "noul",
        "instructions": "Does the customer explicitly threaten to cancel?",
    },
    "is_phishing": {
        "type": "noul",
        "instructions": "Is this email a phishing or scam attempt?",
    },
}

result = agent.predict(state, questions)
print(json.dumps(result["answers"], ensure_ascii=False, indent=2))
```

对于中文业务，当前项目提供了多语言 checkpoint，可以明确加载，而不是默认假设英文底座在中文上同样好用：

```python
agent = laya.load("convaiinnovations/laya-multilingual")
```

这只是选取更合适的起点，不是中文效果保证。领域术语、标签边界和实际数据分布仍然需要单独验证。

## 八、微调实战：把通用接口训练成业务判断能力

到这里，原理已经比较清楚。

真正微调时，最重要的工作不是再写一段更长的提示词，而是把业务规则变成明确的问题、候选项和可信的目标分布。

下面先给出官方复现入口，再提供一个便于改成自有数据的单卡教学方案。两者不要混为一谈。

### 8.1 官方路线：从已有 Notebook 开始

项目提供了双 T4 的微调 Notebook：

```text
notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb
```

它覆盖下载权重、处理 typed-decisions 数据、生成训练脚本、使用 `torchrun` 启动训练、温度校准和导出模型。训练采用编码器与决策头的分组学习率，参考值分别为 `2.5e-5` 与 `1e-4`。

这条路线适合先验证完整链路。不过，当前可见代码有两个值得检查的细节：预处理与训练配置中的输入长度需要保持一致；校准样本虽然已经留出，但按单个问题条目切分，不一定等价于按原始业务案例隔离。

`train_ddp.py` 是 Notebook 单元格生成的文件，不应假设它已经存在于任意工作目录。复现时先执行生成脚本的单元格，再运行启动命令。

### 8.2 自有数据：一个样本应当包含什么

下面是本文教学脚本使用的数据格式。为了便于阅读，示例展开成多行；实际 `.jsonl` 文件中，每个案例写成一行。

```json
{
  "id": "ticket-0001",
  "group_id": "customer-0086",
  "state": {
    "subject": "重复扣费，需要退款",
    "body": "同一笔订单扣了两次款，请今天处理，否则将取消订阅。"
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "这张工单应由哪个部门处理？",
      "criteria": {
        "billing": "支付、账单、重复扣费与退款",
        "technical": "功能故障、系统报错与服务中断",
        "other": "不属于上述类型的问题"
      }
    },
    "urgency": {
      "type": "score",
      "instructions": "按照处理时效判断紧急程度。",
      "criteria": [
        "可以常规排队",
        "需要当日处理",
        "关键业务已中断，需要立即处理"
      ]
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "客户是否明确表示可能取消订阅？"
    }
  },
  "gold": {
    "department": {
      "probabilities": {
        "billing": 1.0,
        "technical": 0.0,
        "other": 0.0
      }
    },
    "urgency": {
      "probabilities": {
        "0": 0.0,
        "1": 1.0,
        "2": 0.0
      }
    },
    "churn_risk": {
      "probabilities": {
        "false": 0.0,
        "true": 1.0
      }
    }
  }
}
```

这是格式示例，不是可用于证明模型效果的训练集。实际标签需要来自业务标注、可核实的后续结果，或者经过说明和质检的教师标注。

`gold.probabilities` 可以是上面这样的 one-hot 硬标签，也可以是软标签，例如多人标注形成的 `[0.1, 0.8, 0.1]`。但不能因为模型需要概率，就凭感觉给每条训练数据编一个小数。

**目标分布从哪里来，决定了模型究竟在学习什么。**

候选顺序尤其容易出错。`choice` 的目标向量要跟 `criteria` 的顺序一致；`score` 必须保留等级顺序；`noul` 在这里采用 `[false, true]`。

打乱候选项时，目标分布必须同步重排。错了这一步，训练仍然会运行，损失也可能下降，却是在教模型学错答案。

### 8.3 切分数据：先分案例，再展开问题

建议准备四份互不重叠的数据：训练集更新权重，验证集选择训练轮次，校准集拟合温度和设计阈值，测试集只做最后评估。

```text
data/
├── train.jsonl
├── valid.jsonl
├── calib.jsonl
└── test.jsonl
```

具体比例不是关键，关键是每份数据覆盖主要类别，并且不能互相泄漏。

切分单位应尽量是客户、对话、原始文档或其他实际关联组，而不是单个问题。先完成案例级切分，再把一份案例展开成部门、紧急程度、流失风险等多个训练条目。

这里还有一个数据设计建议：不要让某个字段成为答案暗号。

例如，所有“需要升级”的样本都有 `history` 字段，而所有“可以自动处理”的样本都没有，那么模型可能只学会检测字段是否存在，并没有理解业务内容。

### 8.4 单卡教学脚本：训练、校准、导出分开处理

随文提供的 `laya_finetune_minimal.py` 是独立编写的教学实现，不是官方 CLI，也不存在本文虚构的 `laya.finetune()` 接口。

它加载现有 checkpoint，更新编码器和决策头，采用：

$$
\mathcal{L}=
\mathcal{L}_{\mathrm{soft\ CE}}
+
\alpha\mathcal{L}_{\mathrm{policy}}
$$

随后在独立校准集上拟合温度。
```python
#!/usr/bin/env python3
"""Teaching implementation: fine-tune Laya on typed decisions, then calibrate.

This is independently written example code, NOT the official Laya trainer.
It uses the public checkpoint loader and the currently exposed model/tokenizer
attributes. Install matching Laya source and dependencies before running.

Input: JSONL case records with id, optional group_id, state, questions, gold.
Keep train/valid/calib/test disjoint at the case/customer/conversation level.
The script validates IDs, group IDs and identical serialized states across splits.

Real checkpoint/GPU training has NOT been executed for this article. --self-test
checks the loss, gradients, calibration and data-validation logic on CPU only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

TYPE_IDS = {"choice": 0, "score": 1, "noul": 2}
TEMP_LO, TEMP_HI = 0.5, 5.0


def json_field(value: Any) -> Any:
    """Accept both nested JSON objects and dataset columns containing JSON text."""
    return json.loads(value) if isinstance(value, str) else value


def read_cases(path: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"id", "state", "questions", "gold"}
            if not isinstance(row, dict) or not required.issubset(row):
                raise ValueError(f"{path}:{line_no}: missing fields {required}")
            row["id"] = str(row["id"])
            row["group_id"] = str(row.get("group_id", row["id"]))
            if row["id"] in seen:
                raise ValueError(f"Duplicate case id in {path}: {row['id']}")
            seen.add(row["id"])
            for field in ("questions", "gold"):
                row[field] = json_field(row[field])
                if not isinstance(row[field], dict) or not row[field]:
                    raise ValueError(f"{path}:{line_no}: {field} must be a nonempty object")
            # Raw text states are legitimate; only decode apparent JSON objects/lists.
            if isinstance(row["state"], str) and row["state"].lstrip().startswith(("{", "[")):
                try:
                    row["state"] = json.loads(row["state"])
                except json.JSONDecodeError:
                    pass
            cases.append(row)
    if not cases:
        raise ValueError(f"No cases in {path}")
    return cases


def check_split_leakage(splits: dict[str, list[dict[str, Any]]]) -> None:
    owners: dict[tuple[str, str], str] = {}
    for name, cases in splits.items():
        for row in cases:
            state = json.dumps(row["state"], ensure_ascii=False, sort_keys=True)
            keys = (("id", row["id"]), ("group", row["group_id"]),
                    ("state", hashlib.sha256(state.encode()).hexdigest()))
            for key in keys:
                prior = owners.setdefault(key, name)
                if prior != name:
                    raise ValueError(f"Data leakage: {key[0]} is shared by {prior} and {name}")


def ordered_target(question: dict[str, Any], gold: dict[str, Any]) -> list[float]:
    qtype = question.get("type")
    criteria = question.get("criteria")
    if qtype == "choice":
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError("choice requires a criteria object with at least 2 options")
        keys = list(criteria)  # Exact insertion order must match option rendering.
    elif qtype == "score":
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError("score requires an ordered criteria list of length >= 2")
        keys = [str(i) for i in range(len(criteria))]
    elif qtype == "noul":
        keys = ["false", "true"]
    else:
        raise ValueError(f"Unknown question type: {qtype}")
    probs = gold.get("probabilities")
    if not isinstance(probs, dict) or set(probs) != set(keys):
        raise ValueError(f"Gold probability keys must exactly match {keys}")
    values = [float(probs[key]) for key in keys]
    if any(not math.isfinite(v) or v < 0 for v in values) or sum(values) <= 0:
        raise ValueError("Targets must be finite, nonnegative and have a positive sum")
    total = sum(values)
    return [v / total for v in values]


def encode_cases(cases, tokenizer, max_len: int, head_max_len: int):
    from laya.common import build_sequence
    items = []
    for row in cases:
        for qid, question in row["questions"].items():
            if qid not in row["gold"]:
                raise ValueError(f"Missing gold for {row['id']}/{qid}")
            if not question.get("instructions"):
                raise ValueError(f"Missing instructions for {row['id']}/{qid}")
            target = ordered_target(question, row["gold"][qid])
            internal = {"t": question["type"], "ins": question["instructions"],
                        "crit": question.get("criteria", {})}
            if "labels" in question:
                internal["labels"] = question["labels"]
            ids, markers = build_sequence(tokenizer, row["state"], internal,
                                          max_len=max_len, head_max_len=head_max_len)
            if len(markers) != len(target):
                raise ValueError(f"Options truncated for {row['id']}/{qid}; increase token budget")
            items.append({"ids": ids, "markers": markers,
                          "qtype": TYPE_IDS[question["type"]], "target": target})
    if not items:
        raise ValueError("No encoded questions")
    return items


def collate(items, pad_id: int):
    batch, length = len(items), max(len(x["ids"]) for x in items)
    kmax = max(len(x["target"]) for x in items)
    ids = torch.full((batch, length), pad_id, dtype=torch.long)
    attention = torch.zeros_like(ids)
    positions = torch.zeros((batch, kmax), dtype=torch.long)
    mask = torch.zeros((batch, kmax), dtype=torch.bool)
    target = torch.zeros((batch, kmax), dtype=torch.float32)
    for i, item in enumerate(items):
        n, k = len(item["ids"]), len(item["target"])
        ids[i, :n] = torch.tensor(item["ids"])
        attention[i, :n] = 1
        positions[i, :k] = torch.tensor(item["markers"])
        mask[i, :k] = True
        target[i, :k] = torch.tensor(item["target"])
    return {"input_ids": ids, "attention_mask": attention, "marker_pos": positions,
            "marker_mask": mask, "qtype": torch.tensor([x["qtype"] for x in items]),
            "target": target}


def forward_model(model, batch):
    return model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                 batch["marker_mask"], batch["qtype"])[0].float()


def distribution_reward(probs, target, mask, qtype):
    """Log + 0.75*spherical - ordinal RPS. The log floor follows the recipe.

    Numerical clipping means this implemented reward should NOT be advertised as
    a proof that finite-data training produces perfectly calibrated probabilities.
    """
    probs = probs * mask
    log_score = (target * probs.clamp_min(1e-12).log().clamp_min(-9.21)).sum(-1)
    spherical = (target * probs).sum(-1) / probs.square().sum(-1).sqrt().clamp_min(1e-9)
    count = mask.sum(-1)
    cdf_error = (probs.cumsum(-1) - target.cumsum(-1)).square()
    indices = torch.arange(probs.size(-1), device=probs.device)
    ordinal_mask = indices < (count - 1).unsqueeze(-1)
    rps = (cdf_error * ordinal_mask).sum(-1) / (count - 1).clamp_min(1)
    return log_score + 0.75 * spherical - torch.where(qtype == 1, rps, 0.0)


def decision_loss(logits, target, mask, qtype, sigma: float, group: int, rl_weight: float):
    logits = logits.masked_fill(~mask, -1e4).float()
    ce = -(target * F.log_softmax(logits, -1)).sum(-1).mean()
    if rl_weight == 0:
        return ce
    b, k = logits.shape
    expanded_mask = mask[:, None, :].expand(b, group, k)
    noise = torch.randn((b, group, k), device=logits.device) * expanded_mask
    noise -= noise.sum(-1, keepdim=True) / expanded_mask.sum(-1, keepdim=True) * expanded_mask
    # A REINFORCE sample must be held fixed when evaluating its log density.
    actions = (logits.detach()[:, None, :] + sigma * noise).masked_fill(~expanded_mask, -1e4)
    probs = actions.softmax(-1)
    with torch.no_grad():
        reward = distribution_reward(probs, target[:, None, :].expand_as(probs),
                                     expanded_mask, qtype[:, None].expand(b, group))
        advantage = reward - reward.mean(-1, keepdim=True)
        advantage /= advantage.std(unbiased=False).clamp_min(1e-6)
    # Density on the zero-sum logit subspace; parameter-independent constants omitted.
    delta = (actions - logits[:, None, :]) * expanded_mask
    log_density = -delta.square().sum(-1) / (2 * sigma * sigma)
    policy = -(advantage * log_density).mean()
    return ce + rl_weight * policy


@torch.no_grad()
def collect_predictions(model, loader, device: torch.device):
    model.eval()
    records = []
    for batch in loader:
        moved = {k: v.to(device) for k, v in batch.items()}
        context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
        with context:
            logits = forward_model(model, moved)
        for i, k in enumerate(batch["marker_mask"].sum(-1).tolist()):
            records.append((int(batch["qtype"][i]), logits[i, :k].float().cpu(),
                            batch["target"][i, :k].float().cpu()))
    return records


def fit_temperatures(records) -> list[float]:
    """Independent bounded grid-search implementation; no training-set fitting."""
    temperatures = []
    grid = torch.exp(torch.linspace(math.log(TEMP_LO), math.log(TEMP_HI), 256)).clamp(TEMP_LO, TEMP_HI)
    grid = torch.sort(torch.cat([grid, torch.ones(1)])).values
    for qtype in range(3):
        rows = [(z, y) for t, z, y in records if t == qtype]
        if not rows:
            temperatures.append(1.0)
            print(f"Warning: no calibration data for question type {qtype}; keeping T=1", file=sys.stderr)
            continue
        scores = torch.zeros_like(grid)
        for z, target in rows:
            scores += -(F.log_softmax(z[None, :] / grid[:, None], -1) * target).sum(-1)
        temperatures.append(float(grid[int(scores.argmin())]))
    return temperatures


def metrics(records, temperatures=None):
    temperatures = temperatures or [1.0, 1.0, 1.0]
    nll, brier, hits, confidence, score_mae = [], [], [], [], []
    for qtype, z, target in records:
        logp = F.log_softmax(z / temperatures[qtype], -1)
        p = logp.exp()
        nll.append(float(-(target * logp).sum()))
        brier.append(float((p - target).square().sum()))
        hits.append(float(p.argmax() == target.argmax()))
        confidence.append(float(p.max()))
        if qtype == 1:
            levels = torch.arange(len(p), dtype=p.dtype)
            score_mae.append(float(abs((p * levels).sum() - (target * levels).sum())))
    if not hits:
        raise ValueError("Empty evaluation set")
    conf, correct = torch.tensor(confidence), torch.tensor(hits)
    bin_ids = (conf * 15).long().clamp(max=14)
    ece = 0.0
    for index in range(15):
        chosen = bin_ids == index
        if chosen.any():
            ece += float(chosen.float().mean() * abs(conf[chosen].mean() - correct[chosen].mean()))
    return {"questions": len(hits), "accuracy_vs_target_argmax": sum(hits) / len(hits),
            "soft_target_nll": sum(nll) / len(nll), "squared_error_to_target": sum(brier) / len(brier),
            "ece_vs_target_argmax": ece,
            "score_expected_value_mae": sum(score_mae) / len(score_mae) if score_mae else None}


def save_checkpoint(model, tokenizer, config, directory: Path):
    from safetensors.torch import save_file
    directory.mkdir(parents=True, exist_ok=True)
    weights = {name: value.detach().cpu().contiguous().clone()
               for name, value in model.state_dict().items()}
    save_file(weights, str(directory / "model.safetensors"))
    model.encoder.config.save_pretrained(str(directory / "encoder"))
    tokenizer.save_pretrained(str(directory / "tokenizer"))
    (directory / "rl_agent_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def self_test():
    torch.manual_seed(19)
    logits = torch.tensor([[1.0, -0.2, 0.1], [-0.1, 0.7, -1e4]], requires_grad=True)
    mask = torch.tensor([[True, True, True], [True, True, False]])
    target = torch.tensor([[0., 1., 0.], [0.25, 0.75, 0.]])
    loss = decision_loss(logits, target, mask, torch.tensor([1, 2]), 0.3, 4, 1.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0 and logits.grad[1, 2] == 0
    assert ordered_target({"type": "noul"}, {"probabilities": {"true": .7, "false": .3}}) == [.3, .7]
    records = [(t, torch.tensor([4., 0.]), torch.tensor([.7, .3])) for t in range(3)]
    temperatures = fit_temperatures(records)
    assert metrics(records, temperatures)["soft_target_nll"] < metrics(records)["soft_target_nll"]
    assert all(TEMP_LO <= t <= TEMP_HI for t in temperatures)
    example = {"id": "one", "group_id": "group", "state": {"text": "same"}}
    try:
        check_split_leakage({"train": [example], "calib": [example]})
    except ValueError:
        pass
    else:
        raise AssertionError("Leakage detection failed")
    print("PASS: target order, finite loss/gradients, masked gradients, temperature search, leakage detection")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", default="convaiinnovations/laya")
    for name in ("train", "valid", "calib", "test"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--out", default="./laya-business")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--head-max-len", type=int, default=192)
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--rl-weight", type=float, default=1.0)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--sigma-start", type=float, default=0.4)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.train, args.valid, args.calib)):
        parser.error("--train, --valid and --calib are required; --test is strongly recommended")
    if min(args.epochs, args.batch_size, args.grad_accum) < 1 or args.group < 2:
        parser.error("epochs/batch-size/grad-accum must be positive; group must be >= 2")
    if min(args.sigma_start, args.sigma_end, args.lr_encoder, args.lr_head) <= 0 or args.rl_weight < 0:
        parser.error("Learning rates and sigmas must be positive; rl-weight must be nonnegative")
    if not 16 <= args.head_max_len < args.max_len:
        parser.error("Require 16 <= head-max-len < max-len")
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is nonempty; use a new --out: {output}")
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("This training example supports CPU/CUDA only (MPS inference is a separate SDK capability)")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits = {name: read_cases(path) for name in ("train", "valid", "calib", "test")
              if (path := getattr(args, name))}
    check_split_leakage(splits)
    import laya
    from safetensors.torch import load_file
    agent = laya.load(args.model, device=str(device))
    if not all(hasattr(agent, name) for name in ("model", "tok", "cfg")):
        raise RuntimeError("Laya SDK attributes changed; review this example against your installed revision")
    model, tokenizer, config = agent.model.float(), agent.tok, dict(agent.cfg)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer has no pad_token_id")
    config.update(max_len=args.max_len, head_max_len=args.head_max_len, temperature=[1., 1., 1.])
    config.pop("temperature_by_options", None)
    if hasattr(model, "temperature"):
        model.temperature.fill_(1.0)
    # No act/escalate supervision is provided: do not train or trust this head here.
    for parameter in model.act_head.parameters():
        parameter.requires_grad_(False)
    if not args.no_checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    loaders = {}
    for name, cases in splits.items():
        items = encode_cases(cases, tokenizer, args.max_len, args.head_max_len)
        loaders[name] = DataLoader(items, batch_size=args.batch_size, shuffle=name == "train",
                                   collate_fn=partial(collate, pad_id=tokenizer.pad_token_id), num_workers=0)
        print(f"{name}: {len(cases)} cases, {len(items)} question sequences")
    encoder_parameters, head_parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            (encoder_parameters if name.startswith("encoder.") else head_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": encoder_parameters, "lr": args.lr_encoder},
        {"params": head_parameters, "lr": args.lr_head}], weight_decay=0.01)
    updates = args.epochs * math.ceil(len(loaders["train"]) / args.grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, history, global_update = math.inf, [], 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        loader = loaders["train"]
        for step, batch in enumerate(loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            progress = global_update / max(updates - 1, 1)
            sigma = args.sigma_start + progress * (args.sigma_end - args.sigma_start)
            chunk_start = step // args.grad_accum * args.grad_accum
            accumulate = min(args.grad_accum, len(loader) - chunk_start)
            context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with context:
                logits = forward_model(model, batch)
                loss = decision_loss(logits, batch["target"], batch["marker_mask"], batch["qtype"],
                                     sigma, args.group, args.rl_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss: inspect targets and lower LR/switch off AMP for diagnosis")
            scaler.scale(loss / accumulate).backward()
            epoch_loss += float(loss.detach())
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:  # No skipped overflow update.
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1
        report = metrics(collect_predictions(model, loaders["valid"], device))
        history.append({"epoch": epoch + 1, "training_loss": epoch_loss / len(loader), "valid": report})
        print(json.dumps(history[-1], ensure_ascii=False))
        if report["soft_target_nll"] < best:
            best = report["soft_target_nll"]
            save_checkpoint(model, tokenizer, config, output)
    model.load_state_dict(load_file(str(output / "model.safetensors"), device=str(device)), strict=True)
    calibration = collect_predictions(model, loaders["calib"], device)
    temperatures = fit_temperatures(calibration)
    config.update(temperature=temperatures, fine_tuned=True, model_name="laya-business-example")
    config.pop("temperature_by_options", None)
    if hasattr(model, "temperature"):
        model.temperature.copy_(torch.tensor(temperatures, device=device, dtype=model.temperature.dtype))
    save_checkpoint(model, tokenizer, config, output)
    report = {"laya_version": getattr(laya, "__version__", "unknown"), "torch_version": torch.__version__,
              "arguments": vars(args), "history": history, "temperatures": temperatures,
              "warning": "Act/escalate head was not trained. Soft targets imply teacher agreement, not ground-truth calibration."}
    if "test" in loaders:
        test_records = collect_predictions(model, loaders["test"], device)
        report["test_uncalibrated"] = metrics(test_records)
        report["test_calibrated"] = metrics(test_records, temperatures)
    (output / "training_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved checkpoint to {output}; temperatures={temperatures}")
    print("No automatic routing threshold has been selected. Validate it on held-out business outcomes.")


if __name__ == "__main__":
    main()

```
脚本会检查跨集合重复的案例 ID、分组 ID 和相同状态文本，校验候选项与标签对应关系，并根据验证集的目标分布 NLL 选择 checkpoint。执行头没有相应标签，因此明确冻结，不作为部署决策依据。

先安装与当前源码一致的环境，并保留版本信息。GPU 训练前应确认 PyTorch 能识别目标设备。

```bash
git clone https://github.com/NandhaKishorM/laya.git

python -m pip install -e ./laya
python -m pip install safetensors

python -c "import torch, laya; print('torch=', torch.__version__); print('laya=', laya.__version__); print('cuda=', torch.cuda.is_available())"

git -C laya rev-parse HEAD
python -m pip freeze > requirements.lock.txt
```

把随文脚本与 `data/` 放在同一个工作目录，以中文业务为例：

```bash
python laya_finetune_minimal.py \
  --model convaiinnovations/laya-multilingual \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --calib data/calib.jsonl \
  --test data/test.jsonl \
  --out ./laya-ticket-model \
  --epochs 4 \
  --batch-size 2 \
  --grad-accum 8 \
  --max-len 1024 \
  --head-max-len 256 \
  --lr-encoder 2.5e-5 \
  --lr-head 1e-4
```

这里的 batch size、训练轮次和学习率只是起步配置，不是某块显卡的显存承诺，也不是已经验证的最优参数。

显存不足时，可以先减小 micro-batch 和输入长度，再观察吞吐与效果。脚本使用梯度累积，但它不能消除单条超长序列本身的显存开销。

**训练前就要设置输入预算。**

假如预处理时已经把正文截成了 512 tokens，后面再把模型配置改成 1024，并不会把被截掉的文本恢复出来。本文脚本在编码数据之前固定这两个值，并在导出时保存相同配置。

脚本还支持一个必要的对照实验：把 `--rl-weight` 设为 `0`，只用软标签交叉熵训练。

```bash
python laya_finetune_minimal.py \
  --model convaiinnovations/laya-multilingual \
  --train data/train.jsonl --valid data/valid.jsonl \
  --calib data/calib.jsonl --test data/test.jsonl \
  --max-len 1024 --head-max-len 256 \
  --rl-weight 0 \
  --out ./laya-ticket-ce-baseline
```

比较两组设置在独立测试集上的准确率、NLL、校准误差和自动处理覆盖率，才能判断额外的策略损失是否真的为当前业务带来收益。



### 8.5 校准不是附赠步骤，要当成独立实验

训练完成后，模型 logits 可能区分得不错，但概率过于尖锐。

温度缩放的形式很简单：

$$
p_k(T)=\operatorname{softmax}(z/T)_k
$$

在同一道题上对所有 logits 使用相同的正温度时，候选排序不变，但分布的尖锐程度会变化。通常 \(T>1\) 会让分布更平缓，\(T<1\) 会让分布更集中。

温度应通过没有参与权重训练的数据拟合，而不是挑一个看起来顺眼的数字。

本文脚本分别为 `choice`、`score`、`noul` 搜索温度，选择让校准集目标分布 NLL 最小的值。搜索范围设置为当前运行时接受的 `0.5～5.0`，避免训练端拟合一个温度，部署端又悄悄截成另一个。

如果采用新的按类型温度，旧 checkpoint 中按候选数量设置的温度覆盖项也需要处理。否则，写进去的类型温度可能没有真正生效。本文脚本在导出时移除旧的 `temperature_by_options`，并保存新温度；后续版本仍应检查实际加载逻辑。

同时要明确：用教师模型软标签拟合温度，主要是在贴近教师分布。

要声称“90% 的预测确实大约有 90% 会正确”，还需要可信的真实结果标签和独立评估。

### 8.6 导出与加载：别只保存一个权重文件

本文脚本的输出目录包含：

```text
laya-ticket-model/
├── model.safetensors
├── rl_agent_config.json
├── encoder/
│   └── config.json
├── tokenizer/
└── training_report.json
```

权重、编码器结构、分词器、输入长度与温度配置共同决定推理行为。只把权重文件复制出去，再随手搭配一个分词器或默认配置，很容易让离线效果与上线结果不一致。

加载自己的 checkpoint 后，用真实测试样本再次检查输出：

```python
import laya

agent = laya.load("./laya-ticket-model")

# state 与 questions 应替换为目标业务的输入。
answers = agent.predict(state, questions)["answers"]
```

本文脚本导出的是可用于推理的 checkpoint，不包含完整优化器、梯度缩放器等训练状态，不应把它当成精确断点续训包。

### 8.7 上线前，别只盯着一个准确率

对于单标签问题，先看准确率、各类别召回率和 Macro-F1，再看 NLL、Brier 分数与可靠性图。

对于 `score`，还应检查等级误差与累计分布误差；对于自动处理策略，则应看：

> 覆盖率提高时，错误率怎样变化？

这些指标回答的问题不同，不能互相代替。

本文脚本为避免混淆，将软标签评测显式命名为 `accuracy_vs_target_argmax`、`soft_target_nll` 等。目标来自教师模型时，这些数字反映的是目标一致性，不应包装成真实业务正确率。

执行阈值也应该是实验结果，而不是从博客复制的常数。以部门分流为例，可以用下面这样的纯业务逻辑表达策略：

```python
def decide_route(answer: dict, threshold: float) -> dict:
    """threshold 必须由独立业务验证确定，不内置所谓安全阈值。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    label = answer["choice"]
    probability = float(answer["probabilities"][label])

    return {
        "department": label,
        "decision": "auto" if probability >= threshold else "review",
        "probability": probability,
    }
```

这段代码刻意读取选中候选项的概率，而不是混用熵置信度与执行头输出。

即便如此，阈值仍然不能替代对分布外输入、关键类别漏判和数据漂移的监控。

## 九、实验结果

### 9.1 保留原始数据，但不把不同测试直接拼成胜负

原文报告，Laya 在单问题推理中达到约 38.4 ms，10 个问题约 156.0 ms，50 个问题约 721.4 ms。
![原文中的 Laya 与 Jev 对比图](https://i-blog.csdnimg.cn/img_convert/a2dac5edb2bdaee73ef5945d09734311.webp?x-oss-process=image/format,png)
这些是作者在其测试条件下的报告值，不是任意机器、任意输入长度的保证。

下面保留原文的主要任务内结果：

| 任务族        |  问题数 N |   准确率 |   ECE |   NLL |
| ---------- | -----: | ----: | ----: | ----: |
| 意图识别与路由    |  1,475 | 99.1% | 0.009 | 0.181 |
| 内容审核与安全    |  2,708 | 96.7% | 0.061 | 0.153 |
| 主题分类       |    749 | 93.9% | 0.029 | 0.196 |
| 情绪与语气      |  1,825 | 90.6% | 0.018 | 0.238 |
| 推断与事实检查    |  3,022 | 88.3% | 0.054 | 0.340 |
| 指令遵循       |    600 | 87.8% | 0.046 | 0.302 |
| 鲁棒性检查      |    744 | 85.1% | 0.108 | 1.058 |
| 阅读理解       |    770 | 84.7% | 0.083 | 0.409 |
| 邮件分流与钓鱼识别  |  2,691 | 73.2% | 0.017 | 0.595 |
| 搜索相关性      |    733 | 62.8% | 0.066 | 0.728 |
| 回答质量评分     |  3,146 | 58.1% | 0.023 | 1.009 |
| 原文标注的任务内总体 | 23,024 | 83.8% | 0.060 | 0.468 |

这里存在一个原文内部的不一致：上面逐项列出的问题数相加为 **18,463**，而总体行写的是 **23,024**。

现有材料不足以解释缺少的是其他任务族，还是汇总口径有误，因此保留原数字，不作猜测性修正。

原文的零样本结果如下：

| 未见任务族   | 问题数 N |   准确率 |   ECE |   NLL |
| ------- | ----: | ----: | ----: | ----: |
| 指令遵循    |   600 | 86.3% | 0.045 | 0.319 |
| 内容审核与安全 |   600 | 79.7% | 0.171 | 1.415 |
| 情绪与语气   |   600 | 58.3% | 0.318 | 1.976 |
| 情感与评分   |   600 | 36.2% | 0.291 | 1.798 |
| 宏平均     | 2,400 | 65.1% | 0.207 | 1.377 |

这些数字更值得关注的地方，不是最高准确率，而是任务间差异：同一个模型，在路由任务上表现不错，不代表它在质量评分或未见情感任务上同样可靠。

原文还报告，在只接受最有信心的部分预测时，接受全部样本的准确率为 83.8%，接受最高置信度的 80% 样本时为 89.4%，接受最高置信度的 50% 时为 92.2%。

这是选择性预测的结果，不意味着任何业务把阈值设为 `0.85`，都能自动处理一半数据并保持 92% 正确率。

### 9.2 当前公开评测也提醒：微调有用，但不能跳过校准

在另一套公开的 typed-decisions 测试中，项目报告原始英文 checkpoint 的准确率约为 36%，微调版本约为 76.6%，后者 ECE 仍约为 0.213。

这个结果与原文不是同一套评测口径，不能直接并排解释为模型变强或变弱；但它说明了一个现实问题：

**适配目标任务可能很重要，而准确率提升也不等于概率已经校准。**([GitHub][13])

![](https://i-blog.csdnimg.cn/direct/2192234ea8cd4c6e9f7d3d5faad3a8bd.png)


*图 3：项目方整理的综合对比图，来源：Laya GitHub 仓库。图中涉及不同来源的 Jev 数据，不属于本文在同机、同数据、同服务条件下完成的配对实验。图中的自托管“零成本”，也不应理解为没有硬件、电力和运维成本。*

原文将自托管模型与外部 API 的延迟、任务内准确率与其他工作流的准确率直接作比，由此得出的“快多少倍”“整体高多少个百分点”，不宜作为严谨结论照搬。

网络传输、批处理策略、文本长度、硬件、训练数据是否覆盖任务，都会影响比较是否成立。当前项目评测文档也明确提醒，外部 Jev 数据并非同一套本地配对实验。

### 9.3 一个更值得借鉴的案例：输入格式可能比继续加数据更关键

项目文档还收录了浏览器动作决策的微调案例。其中一个关键经验是：把页面元素全部塞在 `state` 的 JSON 中，可能导致目标元素被截断；把候选元素及其属性移入选项列表，并为候选部分分配足够 token 预算，效果才明显改善。

这个案例不必当成通用成绩单，值得迁移的是排查方法：

**模型选错了候选项，先确认候选项是否完整进入了模型，而不是直接断言需要更大的模型或更多训练轮次。**

## 十、总结

> 真正值得学习的，是把“生成”和“判断”拆开

Laya 提供的一个有用思路是：面对明确、有限、重复出现的判断任务，没必要总是让生成式大模型先写出答案，再从文本里提取结果。

把状态、问题和候选项一起编码，直接输出分布，可以形成一种更直接的决策接口。

但它的价值不在于给分类器换上“System 1”的名字，也不在于把“经过 RL 训练”当成可信概率的证明。

真正落地时，需要把几件事连起来：

**明确任务边界，准备可信标签，确保输入没有被错误截断，用目标业务数据微调，再用独立数据校准概率和确定升级策略。**

一个小模型把边界清楚的任务做稳，遇到拿不准的情况交给更合适的系统或人工处理，比输出一个看起来很自信、却没有验证过的小数，更有实际价值。

