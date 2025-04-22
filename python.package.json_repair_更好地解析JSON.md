---
id: 0sm3x4unla344rcpajacwcp
title: json_repair_更好地解析JSON
desc: ''
updated: 1745331490381
created: 1745331071770
---

用法类似 json 库，包的 API 如下：

```py
def loads(
    json_str: str,
    skip_json_loads: bool = False,
    logging: bool = False,
) -> Union[JSONReturnType, Tuple[JSONReturnType, List[Dict[str, str]]]]：
    ...

```

例子：

```py
import json_repair
# 单引号、尾随逗号
invalid_json = "{'name': 'Alice', 'age': 30,}"
repaired = json_repair.repair(invalid_json)  # 返回 '{"name": "Alice", "age": 30}'
json.loads(repaired)  # 成功解析
```

**json 库**：仅支持符合 JSON 规范的数据（双引号、无注释、无尾随逗号等）。例如，下面解析会失败：

```py
# 单引号、尾随逗号
invalid_json = "{'name': 'Alice', 'age': 30,}"
json.loads(invalid_json)  # 抛出 JSONDecodeError
```

有时候，使用 json 库解析 LLM 响应的 JSON 内容，可以发现格式错误，对生成的内容提出更高要求。

**json_repair 库**：支持不符合 JSON 规范的数据（单引号、尾随逗号等）。
- 单引号 → 双引号
- 删除尾随逗号
- 移除注释（如 /* comment */）
- 补全缺失的引号或括号

## Ref and Tag