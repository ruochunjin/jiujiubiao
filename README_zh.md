# XuanJi 璇玑

> 一幅璇玑图，读出千首诗。苏蕙的《璇玑图》以 841 字织成方格，可读出数千首诗——一张表，万千读法。**璇玑**对您的数据做同样的事：一张表，万千问答。

[English README](README.md)

璇玑是**表格与数据库的统一问答工具箱**。一条流水线覆盖完整链路，各阶段均可插拔：

```text
问题 + 表格/数据库
  → [1] 消歧（可选，CLEAR 风格）：歧义检测 → 澄清 → 重写
  → [2] 求解 —— 两个可互换引擎：
        • Solver：ReAct 智能体 + 自一致性投票，基于可插拔工具注册表
          （长表、不规范格式、数据库）            [TALON/RAISR 一脉]
        • Refiner：初始化 → 自我批评 → 多工具精化 → 多工具一致性投票
                                                   [TableCritic 一脉]
  → [3] 输出检查：归一化、拒答识别、格式门禁
```

## 快速开始

```python
import xuanji

engine = xuanji.XuanJi()  # 端点由环境变量配置（见下）
result = engine.ask("这位运动员在哪一站获得第一名？",
                    table=[["Year","Venue","Position"],["2003","Paris","1st"]])
print(result.answer)  # Paris
```

数据库一等支持（Solver 自带 schema 感知工具并真实执行 SQL）：

```python
from xuanji import Database
db = Database.from_sqlite("concert_singer.db")  # 或 Database.from_dataframe(df)
result = engine.ask("How many singers are from France?", db=db)
```

**离线开发/测试**无需任何端点——注入 `FakeLLM` 即可跑通全流水线。

## 端点配置

不依赖任何 SDK，纯 HTTP 调用 OpenAI 兼容端点（vLLM / Ollama / DeepSeek / OpenAI 均可）：
`XUANJI_BASE_URL` / `XUANJI_API_KEY` / `XUANJI_MODEL`（兼容 `OPENAI_*` 变量）。

## 命令行

```bash
xuanji --dataset data.json --task tqa --limit 10 --output results.json
```

## 扩展

用 `@register_tool` 注册求解工具（提示词与调度自动更新）；用 `@register_refine_tool` 注册精化工具（generate + critic 接口）。训练侧提供 CORES 过程奖励（格式门禁 + Schema 接地 + 结构相似 + 执行正确），见 `xuanji.training`。

## 学术脉络

璇玑整合了本课题组五篇论文：**CLEAR**（ICDE 2025，消歧）、**TALON**（EMNLP 2025，长表多智能体探索）、**RAISR**（ICASSP 2026，协作数据库智能体与输出检查）、**TableCritic**（CogSci 2025，自我批评 + 工具库）、**CORES**（ACL 2026 Findings，代码中枢推理与过程奖励）。

## 许可

Apache-2.0。移植模块保留原有声明（求解提示词与执行沙箱：Apache-2.0 Google Research；精化模块：CC-BY TableCritic）。
