# JiuJiuBiao 九九表

[![ci](https://github.com/ruochunjin/jiujiubiao/actions/workflows/ci.yml/badge.svg)](https://github.com/ruochunjin/jiujiubiao/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/jiujiubiao)](https://pypi.org/p/jiujiubiao)

> 三七——二十一。每个中国孩子启蒙的第一张表：九九乘法表，有问必答、脱口而出。**九九表**对您的数据做同样的事：一张表，万千问答。

[English README](README.md)

九九表是**表格与数据库的统一问答工具箱**。一条流水线覆盖完整链路，各阶段均可插拔：

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
import jiujiubiao

engine = jiujiubiao.JiuJiuBiao()  # 端点由环境变量配置（见下）
result = engine.ask("这位运动员在哪一站获得第一名？",
                    table=[["Year","Venue","Position"],["2003","Paris","1st"]])
print(result.answer)  # Paris
```

数据库一等支持（Solver 自带 schema 感知工具并真实执行 SQL）：

```python
from jiujiubiao import Database
db = Database.from_sqlite("concert_singer.db")  # 或 Database.from_dataframe(df)
result = engine.ask("How many singers are from France?", db=db)
```

**离线开发/测试**无需任何端点——注入 `FakeLLM` 即可跑通全流水线。

## 端点配置

不依赖任何 SDK，纯 HTTP 调用 OpenAI 兼容端点（vLLM / Ollama / DeepSeek / OpenAI 均可）：
`JIUJIUBIAO_BASE_URL` / `JIUJIUBIAO_API_KEY` / `JIUJIUBIAO_MODEL`（兼容 `OPENAI_*` 变量）。

## 基准与消歧建议

公开基准（K3 端点，种子分层抽样）：WTQ-200 = **0.79**，TabFact-100 = **0.97**。
消歧 A/B（CLAMBSQL 弱类型 50 例）：直接求解 0.695 → 消歧重写 **0.723**；
分类型看，vague 型 0.546 → **0.692**（重写显著帮助），table 型 0.844 → 0.755
（重写可能过度锚定）。**建议**：自由表述的模糊问题开启 `disambiguate=True`；
问题已点名具体 schema 元素时保持关闭。

## 命令行

```bash
jiujiubiao --dataset data.json --task tqa --limit 10 --output results.json
```

## 扩展

用 `@register_tool` 注册求解工具（提示词与调度自动更新）；用 `@register_refine_tool` 注册精化工具（generate + critic 接口）。训练侧提供 CORES 过程奖励（格式门禁 + Schema 接地 + 结构相似 + 执行正确），见 `jiujiubiao.training`。

## 学术脉络

九九表整合了本课题组五篇论文：**CLEAR**（ICDE 2025，消歧）、**TALON**（EMNLP 2025，长表多智能体探索）、**RAISR**（ICASSP 2026，协作数据库智能体与输出检查）、**TableCritic**（CogSci 2025，自我批评 + 工具库）、**CORES**（ACL 2026 Findings，代码中枢推理与过程奖励）。

## 引用本工作

如果您在工作中使用了九九表，请引用相关论文（另见 CITATION.cff）：

```bibtex
@inproceedings{zhang2025clear,
  title={Clear: A parser-independent disambiguation framework for nl2sql},
  author={Zhang, Meng and Ma, Kexin and Xu, Liyang and Zhang, Kedi and Peng, Yuanxi and Jin, Ruochun},
  booktitle={2025 IEEE 41st International Conference on Data Engineering (ICDE)},
  pages={1--14},
  year={2025},
  organization={IEEE}
}

@inproceedings{jin2025talon,
  title={TALON: A multi-agent framework for long-table exploration and question answering},
  author={Jin, Ruochun and Wang, Xiyue and Wang, Dong and Zheng, Haoqi and Qi, Yunpeng and Yang, Silin and Zhang, Meng},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={27385--27401},
  year={2025}
}

@inproceedings{zhang2026reliable,
  title={Reliable Database Question Answering with Collaborative Agents},
  author={Zhang, Meng and Wang, Xiyue and Peng, Yuanxi and Yang, Wenjing and Yang, Silin and Jin, Ruochun},
  booktitle={ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={17142--17146},
  year={2026},
  organization={IEEE}
}

@inproceedings{jin2025tablecritic,
  title={TableCritic: Refine Table Reasoning via Self-Criticism and Tool Library},
  author={Jin, Ruochun and Wang, Dong and Wang, Xiyue and Zheng, Haoqi},
  booktitle={Proceedings of the Annual Meeting of the Cognitive Science Society},
  volume={47},
  year={2025}
}

@inproceedings{zhang2026cores,
  title={CORES: Code-Oriented Reasoning for Complex Text-to-SQL and Generalizable TableQA},
  author={Zhang, Meng and Jin, Ruochun and Peng, Yuanxi and Yang, Wenjing and Wang, Haotian and Sun, Liting and Hu, Kun and Yang, Silin and Ke-di, Zhang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={4827--4847},
  year={2026}
}
```

## 许可

Apache-2.0。移植模块保留原有声明（求解提示词与执行沙箱：Apache-2.0 Google Research；精化模块：CC-BY TableCritic）。
