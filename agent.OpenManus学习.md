---
id: hcawzqs5kib9vt4l1gpqclj
title: OpenManus学习
desc: ''
updated: 1742030211205
created: 1741973130080
---

此项目架构清晰，代码优秀，值得学习。能够在 Manus 公布的几小时之后快速实现，证明此架构能够适应快速开发和试错。参考 [Github](https://github.com/mannaandpoem/OpenManus)

## 一、项目背景与概述

Open Manus 是 MetaGPT 团队复现 Manus 的一个开源项目。Manus 是一款基于大型语言模型（LLM）的应用，由于体验 Manus 需要邀请码且难以获取，MetaGPT 团队在短时间内开发了 Open Manus，旨在提供一个可供研究和学习的开源替代品。

Open Manus 项目在发布后迅速获得了广泛关注，其代码更新频率较高。本次解读基于 2024 年 3 月 9 日下午 2 点的代码版本。

## 二、基础知识：Reactor 模式

Open Manus 实现的是一种 Reactor 模式的单 Agent 系统。理解 Reactor 模式对于理解 Open Manus 的工作原理至关重要。

Reactor 模式包含两个核心要素：Reason（推理）和 Action（行动）。其基本流程如下：

1. **用户输入 (Query):** 用户提出一个问题或指令。
2. **模型思考 (Think):** 模型对用户输入进行推理，确定需要执行的操作。
3. **行动执行 (Action/Function Call/Tool Call):** 模型选择一个工具或函数，并提供相应的参数。
4. **环境/观察 (Environment/Observation):** 执行选定的行动，并将结果反馈给模型。
5. **最终答案 (Final Answer):** 模型基于思考、行动和观察结果，生成最终的答复。

该过程可以循环进行，直到模型认为任务完成并给出最终答案。

## 三、Open Manus 架构与运行模式

Open Manus 目前有两种运行模式：

1. **`python main` (单 Agent 模式):** 只有一个 Manus Agent，负责接收用户输入、选择工具、执行操作并返回结果。
2. **`python run_flow` (双 Agent 模式):** 包含两个 Agent：
  *  **Planning Agent:** 负责生成任务清单 (Checklist)，将复杂任务分解为多个子任务。
  *  **Manus Agent:** 负责执行 Planning Agent 生成的每个子任务。

### 3.1 单 Agent 模式 (`python main`)

用户输入直接传递给 Manus Agent，Agent 决定调用哪些工具（如 Python 代码执行、Google 搜索等），执行工具后将结果返回给 Manus Agent，最终生成并返回结果给用户。

### 3.2 双 Agent 模式 (`python run_flow`)

1. 用户输入传递给 Planning Agent。
2. Planning Agent 生成一个任务清单 (Checklist)，包含多个待办事项。
3. 针对 Checklist 中的每个任务：
  *  Manus Agent 执行任务。
  *  Manus Agent 将执行结果返回给 Planning Agent。
  *  Planning Agent 更新 Checklist，标记已完成的任务。
4. 所有任务完成后，Planning Agent 将最终结果返回给用户。

## 四、代码结构与模块分析

Open Manus 项目主要包含以下几个部分：

### 4.1 `main.py` 和 `run_flow.py`

*  `main.py`: 单 Agent 模式的入口。
*  `run_flow.py`: 双 Agent 模式的入口。

### 4.2 `open_Manus` 目录

*  **`agents`:** 定义了各种 Agent，其中最重要的是：
  *  `MamusAgent`: 继承自 `ToolCallingAgent`，是单 Agent 模式下的主要 Agent。
  *  `PlanningAgent`: 用于双 Agent 模式，负责任务规划。
*  **`flows`:** 包含双 Agent 模式 (`run_flow.py`) 的相关逻辑，单 Agent 模式下不使用。
*  **`prompts`:** 定义了每个 Agent 的提示信息，包括：
  *  **System Prompt:** 描述 Agent 的角色和职责。
  *  **Next Step Prompt (User Instruction):** 指示 Agent 下一步要做什么。
*  **`tools`:** 定义了 Agent 可以使用的各种工具，例如：
  *  `python_code_executor.py`: 执行 Python 代码。
  *  `google_search.py`: 进行 Google 搜索。
  *  `browser.py`: 模拟浏览器操作。
  *  `file_writer.py`: 保存文件。
  *  `finish.py`: 终止流程。

  每个 Agent 可以使用不同的工具组合。Manus Agent 可以使用上述五个工具。

## 五、代码执行流程 (以 `main.py` 为例)

### 5.1 初始化

*  创建 `ManusAgent` 对象。
*  Agent 对象包含：

  *  `prompt`: Agent 的提示信息。

  *  `allowed_tools`: Agent 可以使用的工具列表。



### 5.2 循环执行



1. **接收用户输入:** 等待用户输入下一条指令。

2. **Agent.Run:** 调用 Agent 的 `run` 方法。

  * `run` 方法内部调用 `step` 方法。

3. **Step:** 执行单个步骤，包括：

  *  **Think:** 模型思考，决定下一步行动。

    *  获取 Next Step Prompt (用户指令)。

    *  结合 System Prompt。

    *  调用 `client.chat.completions.create` API (底层使用 LLM) 生成思考结果 (Action/Function Call)。

  *  **Act:** 根据思考结果执行相应的工具。

    *  解析思考结果中的 JSON 或 Function Call 信息。

    *  调用相应的工具函数。

    *  将工具执行结果 (Observation) 记录下来。

  *  **更新记忆 (Update Memory):** 将思考结果和工具执行结果添加到 Agent 的历史消息 (History Message) 中。

4. **判断是否终止:** 如果模型认为任务已完成，则调用 `finish.py` 终止流程。

5. **返回结果:** 将最终结果返回给用户。

6. **循环:** 回到步骤 1，等待下一条指令。



### 5.3 `ToolCallingAgent` 与 `ReactAgent`



*  `ManusAgent` 继承自 `ToolCallingAgent`。

*  `ToolCallingAgent` 实现了 React 模式的具体逻辑。

*  `ReactAgent` 定义了基本的 `run` 和 `step` 方法，实现 Think-Act-Observe 的循环过程。



### 5.4 工具执行 (`execute_tool`)



*  解析 Action/Function Call 中的 JSON 数据。

*  根据解析结果调用相应的工具函数。

*  将工具执行结果作为 Observation 返回。

*  将 Observation 添加到 Agent 的历史消息中。



## 六、双 Agent 模式 (`run_flow.py`) 流程简述



1. **初始化 Planning Agent:** 创建 `PlanningAgent` 对象。

2. **生成 Checklist:** Planning Agent 根据用户输入生成任务清单。

3. **循环执行 Checklist 中的每个任务:**

  *  获取当前步骤 (Step)。

  *  确定执行者 (Executor)，始终为 `ManusAgent`。

  *  `ManusAgent` 执行任务，使用其可用的工具。

  *  `ManusAgent` 将执行结果返回给 `PlanningAgent`。

  *  `PlanningAgent` 更新 Checklist 和状态。

4. **判断是否终止:** 如果 `ManusAgent` 认为任务完成，则触发终止流程。

5. **返回结果:** `PlanningAgent` 将最终结果返回给用户。

双 Agent 模式需要模型具备较强的规划能力。

## 总结

Open Manus 项目提供了一个学习和研究基于 LLM 的 Agent 系统的良好范例。其代码结构清晰，模块化设计良好，易于理解和扩展。通过对 Open Manus 源代码的深入分析，可以掌握 Reactor 模式、Agent 设计、工具调用等关键概念，并了解如何构建一个基于 LLM 的智能 Agent 系统。

## Ref and Tag

[B 站：OpenManus 源代码解读和学习，manus 用不了，那就自己实现一个](https://www.bilibili.com/video/BV1SrRhYmEgm/?share_source=copy_web&vd_source=fe1db0b3f565b5940b244b9c7f7b4fa1)