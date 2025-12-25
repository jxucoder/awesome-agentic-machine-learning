# Awesome Agentic ML [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of awesome agentic machine learning projects, frameworks, and resources.

Agentic ML refers to autonomous AI systems that can plan, execute, and iterate on machine learning workflows with minimal human intervention—from data preprocessing to model training, evaluation, and deployment.

🤖 *This resource list is maintained with the help of [Claude Opus 4.5](https://www.anthropic.com/claude).*

---

## Contents

- [Frameworks & Platforms](#frameworks--platforms)
- [AutoML Agents](#automl-agents)
- [Research Papers](#research-papers)
  - [Multi-Agent AutoML Systems](#multi-agent-automl-systems)
  - [Search & Planning Methods](#search--planning-methods)
  - [Domain-Specific Agentic ML](#domain-specific-agentic-ml)
  - [LLM-Based ML Optimization](#llm-based-ml-optimization)
  - [Foundation Models for ML](#foundation-models-for-ml)
- [Tutorials & Guides](#tutorials--guides)
- [Datasets & Benchmarks](#datasets--benchmarks)
- [Contributing](#contributing)

---

## Frameworks & Platforms

*End-to-end platforms and frameworks for building agentic ML systems.*

| Project | Description | Stars |
|---------|-------------|-------|
| [K-Dense Web](https://k-dense.ai/) | Autonomous AI Scientist platform with dual-loop multi-agent system for deep research, coding, execution, and ML. Surpasses GPT-5 on BixBench by 27%. | - |
| [Karpathy](https://github.com/K-Dense-AI/karpathy) | An agentic Machine Learning Engineer that trains state-of-the-art ML models using Claude Code SDK and Google ADK. By K-Dense. | ![GitHub stars](https://img.shields.io/github/stars/K-Dense-AI/karpathy?style=flat-square) |
| [AutoGluon](https://github.com/autogluon/autogluon) | Open-source AutoML toolkit by Amazon with foundational models and LLM agents for state-of-the-art performance across diverse ML tasks. [NeurIPS 2024 Workshop](https://neurips.cc/virtual/2024/expo-workshop/100328) | ![GitHub stars](https://img.shields.io/github/stars/autogluon/autogluon?style=flat-square) |

---

## AutoML Agents

*Multi-agent systems and LLM-powered agents for automated machine learning pipelines.*

| Project | Description | Stars |
|---------|-------------|-------|
| [AutoML-Agent](https://github.com/DeepAuto-AI/automl-agent) | A Multi-Agent LLM Framework for Full-Pipeline AutoML. Accepted at ICML 2025. | ![GitHub stars](https://img.shields.io/github/stars/DeepAuto-AI/automl-agent?style=flat-square) |
| [AutoGluon Assistant (MLZero)](https://github.com/autogluon/autogluon-assistant) | Multi-agent system for end-to-end multimodal ML automation with zero human intervention. NeurIPS 2025. | ![GitHub stars](https://img.shields.io/github/stars/autogluon/autogluon-assistant?style=flat-square) |

---

## Research Papers

*Academic papers on agentic ML, autonomous ML systems, and LLM-based ML agents.*

### Multi-Agent AutoML Systems

*Frameworks using multiple specialized agents for end-to-end ML pipelines.*

- **AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML** (ICML 2025) - [Paper](https://openreview.net/forum?id=p1UBWkOvZm) | [Code](https://github.com/DeepAuto-AI/automl-agent)  
  Multi-agent system with data, model, and operation agents for full-pipeline automation across tabular, CV, NLP, and time-series tasks.

- **MLZero: A Multi-Agent System for End-to-end Machine Learning Automation** (NeurIPS 2025) - [Paper](https://arxiv.org/abs/2505.13941) | [Code](https://github.com/autogluon/autogluon-assistant)  
  Transforms raw multimodal data into high-quality ML solutions with zero human intervention. Supports CLI, WebUI, and MCP interfaces.

- **LightAutoDS-Tab: Multi-AutoML Agentic System for Tabular Data** (2025) - [Paper](https://arxiv.org/abs/2507.13413) | [Code](https://github.com/sb-ai-lab/LADS)  
  Combines LLM-based code generation with multiple AutoML tools (AutoGluon, LightAutoML, FEDOT) for flexible tabular data pipelines.

- **SmartDS-Solver: Agentic AI for Vertical Domain Problem Solving in Data Science** (ICLR 2026 Submission) - [Paper](https://openreview.net/forum?id=r7gmePFADZ)  
  Reasoning-centric system with SARTE algorithm achieving 81.8% win rate over AIDE+o1-preview on MLE-Bench.

### Search & Planning Methods

*Papers using tree search, MCTS, or structured planning for ML workflow optimization.*

- **I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search** (2025) - [Paper](https://arxiv.org/abs/2502.14693) | [Code](https://github.com/jokieleung/I-MCTS)  
  Introspective node expansion with hybrid LLM-estimated and actual performance rewards. 6% improvement over baselines.

- **PiML: Automated Machine Learning Workflow Optimization using LLM Agents** (AutoML 2025) - [Paper](https://openreview.net/forum?id=Nw1qBpsjZz)  
  Persistent iterative framework with adaptive memory and systematic debugging. 41% submissions above median on MLE-Bench.

### Domain-Specific Agentic ML

*Agentic systems tailored for specific ML domains.*

- **TS-Agent: Structured Agentic Workflows for Financial Time-Series Modeling** (2025) - [Paper](https://arxiv.org/abs/2508.13915)  
  Modular framework for financial forecasting with structured knowledge banks and iterative model selection/refinement.

### LLM-Based ML Optimization

*Using LLMs for specific ML optimization tasks.*

- **Using Large Language Models for Hyperparameter Optimization** (2023) - [Paper](https://arxiv.org/abs/2312.04528)  
  Iterative HPO via LLM prompting. Matches or outperforms Bayesian optimization in limited-budget settings.

### Foundation Models for ML

*Pre-trained models that enable rapid ML development.*

- **TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second** (ICLR 2023) - [Paper](https://arxiv.org/abs/2207.01848) | [Code](https://github.com/automl/TabPFN)  
  Prior-Data Fitted Network using in-context learning for instant tabular classification without training.

---

## Tutorials & Guides

*Tutorials, blog posts, and guides on building agentic ML systems.*

<!-- Add tutorials here -->

---

## Datasets & Benchmarks

*Benchmarks and datasets for evaluating agentic ML systems.*

<!-- Add datasets and benchmarks here -->

---

## Contributing

Contributions are welcome! To add a project or paper, simply [open an issue](../../issues) or submit a PR.

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)

To the extent possible under law, the authors have waived all copyright and related rights to this work.

