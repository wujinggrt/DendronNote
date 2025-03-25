---
id: oojhv04cnkr4q042uc0r9ie
title: OpenAI_API_和数据格式
desc: ''
updated: 1742913132406
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

model: str 必须提供，模型 ID

messages: List[message] 必须提供，array 类型的消息列表。

### system message

预设的系统信息，决定 LLM 扮演的角色，比如 {"role": "assistant", "content": "...", }。必须指定 role, content 为 key

### user message

用户请求信息，必须包含以下字段 role 和 content。

role 必选，在 user message 中应该为 "user"。

content: 必须提供 string 或 array 类型，二选一
- str 类型，标识消息的文本内容，比如 "content": "You are a helpful assistant."
- 数组类型，通常用于多模态模型，通常是一个 json 对象，有文本和图片两个类型：
    1. 文本内容：json 对象，包含 type 和 text 的 key。比如: `{"type": "text", "text": "What is the text in the illustrate?" }`
    2. 图像内容：json 对象，type 和 image_url 作为 key，其中，image_url 需要图像的 URL 或 Base64 编码。比如： `{"type": "image_url", "image_url": {"url": base64_qwen}}`

### assistant message

role 和 content 字段必须提供。

role: str 通常是 "assistant"。

content: str 必须提供，

## Ref and Tag

https://platform.openai.com/docs/api-reference/introduction

OpenAI API格式详解-Chat Completions - 啤酒泡泡的文章 - 知乎
https://zhuanlan.zhihu.com/p/692336625