---
id: vm8q8g8t9xmszwucgowgjdo
title: Pi0_和_pi0fast
desc: ''
updated: 1751810819760
created: 1751739011147
---

## pi0

定义在文件 src/lerobot/policies/pi0/modeling_pi0.py，

## openpi_pytorch

1_e2e_inference.py 文件定义了加载和推理的例子。可以参考观察内容的形状。

```py
observation = {
    "image": {
        "base_0_rgb": torch.randint(
            0, 256, (1, 3, 224, 224), dtype=torch.uint8, device=device
        ),
        # "left_wrist_0_rgb": ...,   Suppose we don't have this view
        # "right_wrist_0_rgb": ...,  Suppose we don't have this view
    },
    "state": torch.randn(1, 8, device=device) * 0.2,
    "prompt": ["do something"],
}

# select action
# let's assume the `action_dim` is 7
action = policy.select_action(observation)[0, :, :7]
```

观察包含图像、机器人的本体感知和 prompt。图像维度是 (b,c,h,w)，本体感知维度是 (b,8)，prompt 维度是 (b,1)。具体 schema 如下：

```py
observation = {
    "image": {
        "base_0_rgb": torch.Tensor, uint8, (bs, c, h, w), # 默认 h=w=224，没有可以不加
        "left_wrist_0_rgb": torch.Tensor, uint8, (bs, c, h, w), # 默认 h=w=224，没有可以不加
        "right_wrist_0_rgb": torch.Tensor, uint8, (bs, c, h, w), # 默认 h=w=224，没有可以不加
    },
    "state": torch.Tensor, float, (bs, state_dim),
    "prompt": List[str], 
    "lang_tokens": torch.Tensor, torch.int64, (bs, seq_len), # 可选
    "lang_masks": torch.Tensor, torch.bool, (bs, seq_len), # 可选
}
```

图像可以使用默认的 224*224，patch size 16，即经典的 ViT 配置。

image 部分，没有可以不提供，内部会自动生成 padding mask。但在 pi0-fast 中，必须全部指定，就算缺失，也要用 zeros 填充。此外，模型没有明确指定必须头部、腕部相机，只是查看了三张图像而已，不必严格遵循 head 和 wrist。

state 部分，需要与模型的 dtype 一致。比如，模型为 bf16 则对应 bf16，float32 则对应 float32。最大维度不超过 32。

### 基座模型 PaliGemma

模型结构为 SigLIP vision encoder 加上 Gemma，在 VQA 数据集上训练对齐。为了用于 VLA 解码 action chunk，pi0 引入一个 300M 的动作专家，把 PaliGemma 的 KV cache 传给动作专家。但是 pi0-fast 不需要动作专家，直接使用 FAST tokenizer 编码。

PaliGemma 输入可分为三部分：
- Image：经过 SigLIP 和投影，得到图像特征，与 Prefix 部分相互可见。Image 和 Prefix 部分使用双向注意力机制，相当于多视角图像的特征直接在时序维度拼接，并没有针对类别处理。
- Prefix：用户输入给 VLM 的 text prompt，描述任务。使用特殊 token `<bos>` 开头，随后是用户输入的文本，最后由特殊 token `<sep>` 结尾，即换行符 "\n"。与 Image 部分相互可见，作为 task instruction。
- Suffix: VLM 的 text 输出部分，在 Prefix 的 `<sep>` 之后的自回归输出，以 `<eos>` 结尾。这部分使用因果注意力机制。在 PI0 中，这部分对应动作专家，输入 state，noisy action，输出 flow matching 的速度场的预测；PI0-FAST 则是 FAST action tokens。

注意：它的词表相比于一般的LLM多了两类特殊词 `<locxxxx>` 和 `<segxxx>`，当你的输入是 "detect [entity]" 的时候，模型会给图片打bounding box找寻 [entity]，bounding box用 4个 `<loc0000> ~ <loc0999>` 来表示 normalized box；当你的输入是 "segment [entity]"，模型会用 `<seg000> ~ <seg0127>` 输出分割。

模型 paligemma 在 p0/paligemma_with_expert.py:PaliGemmaWithExpertModel。

```py
class PaliGemmaWithExpertModel(PreTrainedModel):
    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        # 纯正的 Huggingface Transformers PaliGemma 模型
        self.paligemma = PaliGemmaForConditionalGeneration(
            config=config.paligemma_config
        )
        # 纯正的 Huggingface Transformers Gemma 模型
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)

    # 对于这个类我们几乎只需要关注这一个方法
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        inputs_embeds: List[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        fill_kv_cache: Optional[bool] = None,
    ):
        """
        Args:
            attention_mask (Optional[torch.Tensor], optional): 
                Attention mask with shape (b, seq_len, seq_len). Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): 
                Position indices 用于给QK加RoPE.
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]], optional):
                可选的kv cache
            inputs_embeds (List[torch.FloatTensor], optional): 
                输入 embeddings，包括了image和language，并且都已经完成了embedding
            use_cache (Optional[bool], optional): 
                是否使用kv cache
            fill_kv_cache (Optional[bool], optional): 
                是否将本次的kv value返回作为cache给下次使用

        Returns:
            outputs_embeds (torch.Tensor): 输出embedding
            past_key_values (Optional[Union[List[torch.FloatTensor], Cache]]): 
                本次的kv value，如果fill_kv_cache=False，那么会是空字典
        """
        ......
        # 这个方法没有直接使用PaliGemmaForConditionalGeneration和GemmaForCausalLM，而是读取它们所有的layers
        # 逐层经attention和ffn处理特征，几乎没有什么能自定义的地方
        # 唯一能修改的是attention的计算方式，入口在self.get_attention_interface()
        # 通过修改这个可以引入 flashattention，flexattention，xformers等等优化方案
```

### PI0

参考 pi0/modeling_pi0.py:PI0Policy，负责输入的预处理，包括图像、text prompt，机器人本体感知，动作的预处理，准备后交给 PI0FlowMatching 运算。

```py
# 这部分的代码对应 pi0/modeling_pi0.py 下的 PI0Policy

class PI0Policy(PreTrainedPolicy):
    def __init__(
        self,
        config: PI0Config,
        tokenizer_path: str = "google/paligemma-3b-pt-224",
    ):
        super().__init__(config)
        self.config = config
        # 纯纯的 Huggingface Transformers PaliGemma Tokenizer
        self.language_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # nn.Module，稍后再具体看它
        self.model = PI0FlowMatching(config)
        self.reset()

    # 这是lerobot设计必须实现的函数，意图在episode开始时策略reset
    # 在lerobot的实现中维护了一个action buffer，每次生成50长度的action chunk会压入buffer中
    # 每次select_action都只输出最近的一个action，buffer空了再重新生成
    def reset(self):
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(
        self, observation: dict[str, Tensor], noise: Tensor | None = None
    ):
        """
        最主要的决策函数，输入observation输出action，格式见下，更具体的格式可以看第一篇博客☝ 
        Observation: {
            "image": {
                "base_0_rgb": (*b, c, h, w),  # uint8 [0, 255]
                ...
            },
            "state": float32 [*b, s],
            "prompt": List[str],

            "lang_tokens": float32 [*b, l],
            "lang_masks": float32 [*b, l],
        }
        either provide `prompt` or (`lang_tokens`, `lang_masks`).
        """
        self.eval()
        
        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        lang_tokens, lang_masks = self.prepare_language(observation)
        actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, state, noise=noise
        )
        return actions

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        # 计算loss的函数，输入和select_action一样，需要多给action标签
        ......
        return loss, loss_dict

    def prepare_images(self, observation: dict[str, Tensor]):
        """
        预处理图像，会经过如下操作
        1. normalize到[-1,1]，匹配SigLIP的输入要求
        2. resize到(224,224)
        3. 将图像拼接成形状(b, n, c, h, w)，n表示不同视角，PI0最大支持n=3
        
        输入是 select_action 的输入
        输出包括 1. 形如(b,n,c,h,w)的images，2.形如(b,n)的masks，指示了哪些images提供或未提供
        """
        ......
        return images, img_masks

    def prepare_state(self, observation: dict[str, Tensor]):
        """
        预处理state，会经过如下操作
        1. zero padding到32维度 (b, s_dim) -> (b, 32)
        """
        ......
        return state

    def prepare_action(self, observation: dict[str, Tensor]):
        """
        预处理action，会经过如下操作
        1. zero padding到32维度 (b, a_dim) -> (b, 32)
        """
        ......
        return action, action_dim

    def prepare_language(self, observation: dict[str, Tensor]):
        """
        预处理language，会经过如下操作
        1. 还记得上一节介绍的PaliGemma的输入格式吗？这里就会自动为language加上<bos>和\n的开头结尾
        2. 转化为离散的Token
        """
        ......
        return lang_tokens, lang_masks
```

同文件下，流匹配逻辑：

```py
class PI0FlowMatching(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # paligemma with action expert 这就是第二章我们介绍的基座部分哦
        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_with_export_config
        )
        ......

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理完成prefix部分的embedding
        在外包装PI0Policy中我们对所有模态完成了向量化，那么这里我们就要将各个部分拼装起来了
        可以看到输入要求的变量都是之前预处理得到的输出
        Args:
            images (torch.Tensor):    float (*b, n, c, h, w) images in range [-1.0, 1.0]
            img_masks (torch.Tensor):  bool (*b, n) masks for images
            lang_tokens (torch.Tensor): int (*b, l) language tokens
            lang_masks (torch.Tensor): bool (*b, l) masks for language tokens
         
        这个函数基本上完成了如下操作
        1. 用SigLIP将images编码成embedding (b, num_patch, d)
        2. 将lang_tokens编码成embedding (b, num_tokens, d)
        3. 将这两部分拼装成一个<images> <bos> <task> <sep>的embedding (b, seq_len, d)
        4. 根据img_masks和lang_masks拼装成pad_masks (b, seq_len)
        5. att_masks就是双向可见的mask
        """
        ......
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """
        处理完成prefix部分的embedding
        在外包装PI0Policy中我们对所有模态完成了向量化，那么这里我们就要将各个部分拼装起来了
        可以看到输入要求的变量都是之前预处理得到的输出
        Args:
            state (torch.Tensor):         float32 (*b, s) robot state
            noisy_actions (torch.Tensor): float32 (*b, n, m) noisy actions
            timestep (torch.Tensor):      float32 (*b,) timestep in [0, 1] range

        这个函数基本上完成了如下操作
        1. state, action过线性projection
        2. timestep (flow-matching t) 过time embedding
        3. 拼接好所有的输入得到embs
        4. 由于这些信息都不会有padding，所以pad_masks就是ones
        5. 还记得在第二章我们介绍过suffix是casual attention吗！这里att_masks就是casual的
        """
        ......
        return embs, pad_masks, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        """ 计算loss的function，就是直接的flow matching loss """
        return losses

    def sample_actions(
        self, images, img_masks, lang_tokens, lang_masks, state, noise=None
    ) -> Tensor:
        """
        差不多经过了如下的操作
        1. 对vision-language进行prefix emb，然后用PaliGemma处理得到kv cache，存下来（大运算量）
        2. 进行flow-matching的解噪过程，每次解噪要进行suffix emb，然后用Gemma expert处理，这里面会使用到kv cache (小运算量)
        3. 完成解噪，得到生成的action
        """
        return x_t
```

### 配置

src/lerobot/policies/pi0/configuration_pi0.py 指出：

```py
class PI0Config(PreTrainedConfig):
    # Projector
    proj_width: int = 1024
    # Decoding
    num_steps: int = 10
    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2, flex
    ......
```

此配置传递给 PI0Policy 和 PI0FlowMatching。

#### VLM 和 Expert 的配置

VLM 部分使用 PaliGemma，而动作专家部分是 300M 参数的 GemmaForCausalLM 模型。

### Insights

是否可以自己训练对齐？VLM 就像 SmolVLA 仅用前几层，关注于理解语言和视觉。甚至可以只用 SigLIP 和 Qwen Embedding 来训练动作专家，这样动作专家可以快速推理。再用 VLM，甚至加上声音模态的 MLLMs 规划任务，发布子任务给动作专家执行。需要考虑好对齐的问题。

## Ref and Tag

简单好用的PI0/PI0-fast的PyTorch实现（二） - 董子斌的文章 - 知乎
https://zhuanlan.zhihu.com/p/1920507606037936040