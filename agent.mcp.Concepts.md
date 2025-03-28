---
id: 9gc9g6ffxfzjvj43sh58m28
title: Concepts
desc: ''
updated: 1743182778740
created: 1743147909324
---


## MCP 协议

MCP 是一种开放协议，提供了应用程序向 LLMs 提供上下文的标准。MCP 协议由 Anthropic 在 2024 年 11 月底提出：
- 官方文档：[Introduction](https://link.zhihu.com/?target=https%3A//modelcontextprotocol.io/introduction)
- GitHub 仓库：github.com/modelcontextprotocol

### 为什么需要 MCP

大模型产品有:
- chatbot: 只会聊天，比如 DeepSeek, ChatGPT
- composer: 做具体工作，比如 cursor, copilot
- agent: 私人秘书，输入问题，自动执行，Open Manus

为了实现 Agent，即需要让 LLM 自如灵活地操作软件、物理世界的机器人，需要定义统一的上下文协议和对应的统一工作流。MCP (model context protocol) 是一套基础协议，解决此类问题。感性认识如下：

![感性认识](assets/images/agent.mcp.Concepts/感性认识.png)

### 总体架构

MCP 核心使用客户端-服务器架构，host 可以连接到多个服务器：

![cs](assets/images/agent.mcp.Concepts/cs.png)

- MCP 主机（MCP Hosts）：MCP 主机是指希望通过 MCP 访问数据的程序，例如 Claude Desktop、集成开发环境（IDEs）或其他 AI 工具。
- MCP 客户端（MCP Clients）：MCP 客户端是与服务器保持 1:1 连接的协议客户端，负责与 MCP 服务器通信。
- MCP 服务器（MCP Servers）：MCP 服务器是轻量级程序，每个服务器通过标准化的 Model Context Protocol 暴露特定的功能。
- 本地数据源（Local Data Sources）：本地数据源是指 MCP 服务器可以安全访问的计算机文件、数据库和服务。
- 远程服务（Remote Services）：远程服务是指 MCP 服务器可以通过互联网连接的外部系统（例如通过 API 访问的服务）。

## MCP Server 中的基本概念

### openai 协议

使用 Python 或 Typescript 开发 app 时，通常安装 openai 库。根据模型厂商的 url 和模型类别，我们可以访问大模型。大模型提供商也需要支持此库与协议。以 deepseek 服务为例：

```py
from openai import OpenAI

client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
```

查看 create() 方法，可以发现 openai 协议需要大模型厂商支持众多 feature。比如，常见的有 `temperature`, `top_p`。

一次普通调用涉及众多可调控参数。其中有 `tools` 参数：

```py
@overload
    def create(
        self,
        *,
        messages: Iterable[ChatCompletionMessageParam],
        model: Union[str, ChatModel],
        ...
        tools: Iterable[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
    ) -> ChatCompletion:
```

tools 参数要求大模型厂商必须支持 function calling 特性。我们可以提供部分工具描述（和 MCP 协议完全兼容），在 tools 非空情况下，chat 函数返回值中会包含 `tool_Calls`，描述调用函数的组件：

```py
from openai import OpenAI

client = OpenAI(
    api_key="Deepseek API",
    base_url="https://api.deepseek.com"
)

# 定义 tools（函数/工具列表）
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "获取给定地点的天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市，比如杭州，北京，上海",
                    }
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个很有用的 AI"},
        {"role": "user", "content": "今天杭州的天气是什么？"},
    ],
    tools=tools,  # 传入 tools 参数
    tool_choice="auto",  # 可选：控制是否强制调用某个工具
    stream=False,
)

print(response.choices[0].message)
```

运行返回如下：

```py
ChatCompletionMessage(
    content='',
    refusal=None,
    role='assistant',
    annotations=None,
    audio=None,
    function_call=None,
    tool_calls=[
        ChatCompletionMessageToolCall(
            id='call_0_baeaba2b-739d-40c2-aa6c-1e61c6d7e855',
            function=Function(
                arguments='{"location":"杭州"}',
                name='get_current_weather'
            ),
            type='function',
            index=0
        )
    ]
)
```

tool_calls 给出了大模型使用工具的方法。

openai 协议中，tools 仅支持函数类调用。可以对函数类的调用模拟资源获取等工作。

## MCP Quick Start

MCP 官方提供了封装的 SDK，我们可以快速开发 MCP 服务器。

安装：

```bash
pip install mcp "mcp[cli]" uv
```

```py
# server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP('锦恢的 MCP Server', version="11.45.14")

@mcp.tool(
    name='add',
    description='对两个数字进行实数域的加法'
)
def add(a: int, b: int) -> int:
    return a + b

@mcp.resource(
    uri="greeting://{name}",
    name='greeting',
    description='用于演示的一个资源协议'
)
def get_greeting(name: str) -> str:
    # 访问处理 greeting://{name} 资源访问协议，然后返回
    # 此处方便起见，直接返回一个 Hello，balabala 了
    return f"Hello, {name}!"

@mcp.prompt(
    name='translate',
    description='进行翻译的prompt'
)
def translate(message: str) -> str:
    return f'请将下面的话语翻译成中文：\n\n{message}'
```

## Ref and Tag

Agent 时代基础设施 | MCP 协议介绍 - 锦恢的文章 - 知乎
https://zhuanlan.zhihu.com/p/28859732955

优雅地开发 MCP 服务器（一）MCP 中的 Resources，Prompts，Tools 和基本调试方法 - 锦恢的文章 - 知乎
https://zhuanlan.zhihu.com/p/32593727614