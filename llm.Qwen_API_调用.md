---
id: n2gd1ryopn502cseud5lg4z
title: Qwen_API_调用
desc: ''
updated: 1743173031142
created: 1742980461639
---

base_url: https://dashscope.aliyuncs.com/compatible-mode/v1


可以使用 curl 尝试。如果是 curl，则请求的 url 需要添加 /chat/completions。注意，JSON 中的数组和对象的最后一条不能有逗号。

```bash
curl --request POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer API_KEY" \
  --data '{
    "model": "qwen2.5-vl-32b-instruct",
    "messages": [
        {"role": "system", "content":  [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                    }
                },
                {"type": "text", "text": "Outline the position of dog and output all the coordinates in JSON format."}
            ]
        }
    ],
    "stream": false
  }'
```

## Ref and Tag

[首次调用](https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen?spm=a2c4g.11186623.0.i2)
[API 参考](https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-by-calling-api?accounttraceid=6f8a4553f02d448090083a7c0bd17c00jhcx#2ed5ee7377fum)
[Qwen VL](https://help.aliyun.com/zh/model-studio/user-guide/vision/)