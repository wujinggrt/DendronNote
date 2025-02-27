---
id: k6nf2mx4rvbnk41pu3l84tk
title: 节省显存
desc: ''
updated: 1740637922160
created: 1740637890543
---


## 训练后及时清空 cuda cache

```py
        torch.cuda.empty_cache()
        gc.collect()
        del input_ids
        del attention_mask
        del position_ids
        del past_key_values
        del inputs_embeds
        del labels
        del pixel_values
        del image_grid_thw
        del actions
        del states
```

## Ref and Tag