---
id: oojhv04cnkr4q042uc0r9ie
title: OpenAI_API_和数据格式
desc: ''
updated: 1743095851119
created: 1742897914437
---


兼容 OpenAI 的 API 请求，首先需要指定 OPENAI_API_KEY 和 BASE_URL，首先需要构建 request 请求体：

```bash
curl https://api.openai.com/v1/chat/completions \      <===你要请求的BASE_URL
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \         <===你的OPENAI_API_KEY
  -d '{
    "model": "No models available",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ]
  }'
```

response 如下：

```console
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo-0125",
  "system_fingerprint": "fp_44709d6fcb",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "logprobs": null,
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
```

## request body: 请求消息体的字段和内容

### 字段概览

- model: str 必须提供，模型 ID
- messages: List[dict] 必须提供，数组中的都是 JSON 对象。在 Python 的 OpenAI 相关库中，使用 List 表达。
- tools: List[dict] 可选，提供模型选择的工具列表。目前工具仅支持函数。工具数量最多 128 个。
- tool_choice: Union[str, dict] 可选，str 或 JSON 对象类型，控制模型调用哪个函数。
- stream: bool 可选，true 代表流式地返回消息，就像 ChatGPT 逐字生成。token将在可用时作为data-only的SSE(server-sent events)事件发送给用户，http的chunk流由 data: [DONE] 消息终止。常用于聊天 Agent，快速响应用户。

### messages 的细节

具体 message 数组的 JSON 对象如下。

#### system message

预设的系统信息，决定 LLM 扮演的角色，比如 {"role": "assistant", "content": "...", }。必须指定 role, content 为 key

#### user message

用户请求信息，必须包含以下字段 role 和 content。即使是申请规划工具的使用，即函数调用，通常还是用 "user" 的角色。使用 user message。tool message 较少使用。

role 必选，在 user message 中应该为 "user"。

content: 必须提供 string 或 array 类型，二选一
- str 类型，标识消息的文本内容，比如 "content": "You are a helpful assistant."
- 数组类型，通常用于多模态模型，通常是一个 json 对象，有文本和图片两个类型：
    1. 文本内容：json 对象，包含 type 和 text 的 key。比如: `{"type": "text", "text": "What is the text in the illustrate?" }`
    2. 图像内容：json 对象，type 和 image_url 作为 key，其中，image_url 需要图像的 URL 或 Base64 编码。比如： `{"type": "image_url", "image_url": {"url": base64_qwen}}`

在 OpenManus 中，请求 LLM 决定使用哪些工具时，使用 user message，并且传入 tools 字段。

#### assistant message

向 LLM 请求后给出响应后，需要将响应组织为 assistant message，用作在下次询问时的上下文信息。上下文信息为用户的请求和 LLM 的响应历史。assistant message role 和 content 字段必须提供。通常用于附加在对话历史中。

role: str 通常是 "assistant"。代表 LLM 的角色是 assistant。

content: str 必须提供，提供助手消息的内容。如果在此 assistant message 中，指定 tool_calls时可以不提供。

name: str 可选。参与对话者的名称。

tool_calls: 可选，类型是数组。使用大模型生成合适的工具调用后，即 user message 请求字段带有 tools 后，LLM 返回的响应体包含 tool_calls 字段，可以将其组织到本 tool_calls 字段中，组织为 assistant message，用于提供对话历史。例如：函数调用。如果是数组，则每个元素都是 JSON 对象，各自代表函数调用。JSON 字段要有：
- id：str 必须提供。表示函数调用的id
- type：str 必须提供，表示工具调用的类型。目前仅支持 "function" 类型
- function: str 必须提供，表示模型针对工具调用为用户生成的函数说明，即模型在特定任务和场景下，在用户提供的函数中，会推断出应该使用哪一个函数，以及函数的参数应该是什么。所以包括的字段有：
  - name: str 必须提供，要调用的函数的名称
  - arguments: str 必须提供，表示调用函数所用的参数，由模型以 JSON 格式生成(如："{\n\"location\": \"Boston, MA\"\n}")。但是请注意，模型并不总是生成有效的参数，并且可能会产生未由函数定义的参数。在调用函数之前最好验证参数的准确性。

#### tool message

根据 assistant 的 tool_calls 内容调用了某个函数后，用户可能还需要再把函数调用结果**反馈**给大模型，让大模型根据函数调用结果给出最终的总结性的答复。字段有：

- content: str 必须提供，工具消息内容，描述调用函数后的结果
- role: str 必须提供，作为角色，通常是 "tool"
- tool_call_id: str 必须提供，本次消息对应函数调用的结果反馈。与 assistant message.tool_calls.id 对应。

### tools 字段细节

数组中的 JSON 对象包含字段如下：

type: str 必须提供，工具类型。目前仅支持 "function"

function: 必须提供，是 JSON 对象类型，描述函数信息。包括：
- description: str 可选，描述函数功能，模型用来选择何时调用此函数
- name: str 必选，函数名称
- parameters: JSON 可选，JSON 对象类型，描述函数接受参数，JSON Schema 对象。不包含此字段代表参数为空。

### tool_choice: Agent 常用

传入 str 场景：
- "none" 表示模型不会调用函数而是生成消息。
- "auto" 意味着模型可以在生成消息或调用函数之间进行选择。

当不存在任何函数时，"none" 是默认值。如果存在函数，则 "auto" 是默认值。

传入 JSON 场景是，则是强制模型生成调用此函数的信息。例如，设置为 {"type": "function", "function": {"name": "my_function"}} 指定特定函数来**强制模型调用该函数**。

### 以工具调用为例子

工具调用，在构建 agent-like 这类智能体时，非常的有用，他可以打破模型本身的知识边界。这里举一个大模型使用工具的例子，场景为：用户问某个地方的天气怎么样

```bash
curl https://api.openai.com/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $OPENAI_API_KEY" \
-d '{
  "model": "gpt-4-turbo",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"     <==用户的问题
    }
  ],
  "tools": [                   <===用户提供工具集让模型来选择使用哪个可以解决用户的问题
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}'
```

在 tools[0].function.parameters.type 为 object，代表参数是一个 JSON 对象。

生成可能如下：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699896916,
  "model": "gpt-3.5-turbo-0125",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [                       ===>大模型根据用户的问题和用户提供的工具集返回了可能有用的函数
          {
            "id": "call_abc123",
            "type": "function",
            "function": {
              "name": "get_current_weather",  ===>用户可以在调用了该函数后在下一轮对话把该函数调用结果反馈给大模型，然后就可以得到一个最终的答复
              "arguments": "{\n\"location\": \"Boston, MA\"\n}"
            }
          }
        ]
      },
      "logprobs": null,
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 82,
    "completion_tokens": 17,
    "total_tokens": 99
  }
}
```

## response body: 响应消息体的字段和内容

- id: str 唯一标识
- choices: List[dict] 一个或多个聊天响应列表。请求模型生成多个答复，即请求的参数 n 大于 1 时，列表的元素会是多个的，比如 GRPO 的强化学习训练，生成多个组。一般情况只会有一个。

### choices

每个 JSON 对象包含字段：

- index: int 在 choices 的索引，常为 0
- finish_reason: str 表示模型停止生成token的原因。“stop” 代表模型达到自然停止点或提供的停止符号、“length” 代表已达到请求中指定的最大token数、“content_filter” 代表内容被过滤了、“tool_calls” 代表模型要调用工具
- message: JSON 对象类型，表示模型生成的聊天消息
  
message 的 JSON 对象字段如下：
- content: str 消息内容
- role: str 消息作者角色，比如 assistant
- tool_calls: List[JSON] 工具调用信息，比如函数调用。每个元素使一个 JSON 对象，代表函数调用。

tool_calls 数组的 JSON 对象包含字段：
- id: str 函数调用 id 
- type: str 工具调用类型，仅支持 "function"
- function: JSON 模型针对工具调用为用户生成的函数说明。在特定任务和场景，根据用户提出函数，推断使用的函数和参数。包含如下字段：
  - name: str 调用的函数名称
  - arguments: str 调用的参数，字符串内容为 JSON 格式。但是注意，并不总是生成有效对象。

## Python 中的接口

OpenAI.chat.completions.create() 返回 ChatCompletion 实例。

### ChatCompletion

```py
class Choice(BaseModel):
    finish_reason: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    index: int
    logprobs: Optional[ChoiceLogprobs] = None
    message: ChatCompletionMessage

class ChatCompletion(BaseModel):
    id: str
    choices: List[Choice]
    created: int
    model: str
    object: Literal["chat.completion"]
    service_tier: Optional[Literal["scale", "default"]] = None
    system_fingerprint: Optional[str] = None
    usage: Optional[CompletionUsage] = None
```

一般从 ChatCompletion.choices[0].message 获取对话结果。由于 ChatCompletionMessage 也继承了 Pydantic 的 BaseModel，可以使用 model_dump() 方法来获取其字典格式。

### ChatCompletionMessage: 响应消息体

```py
class FunctionCall(BaseModel):
    arguments: str
    name: str

class ChatCompletionMessage(BaseModel):
    content: Optional[str] = None
    refusal: Optional[str] = None
    role: Literal["assistant"]
    audio: Optional[ChatCompletionAudio] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
```

```py
class Function(BaseModel):
    arguments: str
    name: str
class ChatCompletionMessageToolCall(BaseModel):
    id: str
    function: Function
    type: Literal["function"]
```

## Ref and Tag

https://platform.openai.com/docs/api-reference/introduction

OpenAI API格式详解-Chat Completions - 啤酒泡泡的文章 - 知乎
https://zhuanlan.zhihu.com/p/692336625

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))