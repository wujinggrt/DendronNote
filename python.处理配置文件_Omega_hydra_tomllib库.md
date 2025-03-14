---
id: qb5cah4vrkew5znsyd9u9ur
title: 处理配置文件_Omega_hydra_tomllib库
desc: ''
updated: 1741972242589
created: 1741869359576
---

## OmegaConf：读取和操作 yaml，转为字典对象

OmegaConf是配置管理库，读取和操作yaml配置文件，并将其转换为字典对象。并且提供默认值等额外功能。pip安装即可。

```yaml
# config.yaml
database:
  host: localhost
  port: 3306
  user: admin
  password: secret
```

在 Python 中，使用如下方式加载和访问配置项。

```py
from omegaconf import OmegaConf
# 从文件加载配置
cfg = OmegaConf.load('config.yaml')
# 访问配置项
print(cfg.database.host)
# 修改配置项
cfg.database.host = 'newhost'
# 打印整个配置，但是原文件不会修改
print(OmegaConf.to_yaml(cfg))
```

## Hydra：基于 OmegaConfg 的高级配置库

安装使用 pip install hydra-core 命令即可。Hydra 提供更加灵活的方式处理配置文件。比如，访问如下：

```yaml
class_name: MyClass
params:
  param1: value1
  param2: 42
  param3:
    - item1
    - item2
```

```py
from hydra import initialize, compose
from MyClass import MyClass  # 导入你定义的类

if __name__ == "__main__":
    with initialize(config_path="."):
        cfg = compose(config_name="config")
        
        # 根据配置文件实例化类
        my_instance = MyClass(**cfg.params)

        # 调用类的方法
        my_instance.print_params()
```

### 配置文件中引用其他键对应的值

原生 YAML 不支持，仅仅提供引用和锚点。但是 hydra 可以使用 `${变量名}` 的形式访问。例如：

```yaml
  horizon: ${horizon}
```

### 插值

在脚本中，进行简单计算需要定义插值处理器。比如，在 Python 中的 eval() 函数，处理 `eval:'EXPR'` 中 EXPR 的内容：

```yaml
  pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
```

一般在训练脚本的 main() 中，或者全局范围，注册 resolver：

```python
# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()
```

main 中提到的 ${now:} 是配置中用于 log 的部分：
```yaml
  name: ${now:%Y.%m.%d-%H.%M.%S}_${name}_${task_name}
  tags: ["${name}", "${task_name}", "${exp_name}"]
```

### 实例化：依赖注入，IoC

类似 Spring 的 context，hydra 使用 `_target_` 实现依赖注入。我们可以指定一个类或函数的完整路径，以便在配置中实例化该类或调用该函数。这是一种配置驱动的实例化机制，允许你在配置文件中描述你的程序组件应该如何被创建，而无需在代码中硬编码具体的类实例化。

`_target_` 指定实例化的具体类。顶层的 `_target_` 代表整个配置树都用于实例化一个单一的对象。这对于定义复杂对象的构造特别有用，可以在子树继续嵌套地指定 `_target_` 来实例化，用于传给顶层配置指定的类，作为构造器的参数。具体实例化使用 hydra.utils.instantiate(cfg.model) 的形式调用。

```py
model:
  _target_: my_package.models.MyModel
  param1: 42
  param2: "some_value"
```

```py
from hydra import initialize, compose

if __name__ == "__main__":
    with initialize(version_base=None, config_path=".", config_name="config"):
        cfg = compose(config_name="config")

        # Hydra会根据配置文件自动实例化model
        model = hydra.utils.instantiate(cfg.model)

        # 接下来你可以使用model实例
        model.do_something()
```

子树配置使用 `_target_` 如下：

```yaml
policy:
  _target_: diffusion_policy.policy.diffusion_transformer_lowdim_policy.DiffusionTransformerLowdimPolicy
  model:
    _target_: diffusion_policy.model.diffusion.transformer_for_diffusion.TransformerForDiffusion
    input_dim: ${eval:'${action_dim} if ${obs_as_cond} else ${obs_dim} + ${action_dim}'}
    output_dim: ${policy.model.input_dim}
```

我们也可以用引用的方式：

```py
db:
  _target_: my_package.database.Database
  connection_string: "sqlite:///my_database.db"
app:
  _target_: my_package.app.Application
  db: ${db}
logger:
  _target_: my_package.logger.Logger
  db: ${db}
```

它们共享一个 db 对象实例，不会创建两个 db。这是因为 Hydra 在解析配置并实例化对象时，遵循以下原则：
1. 单一实例：相同的配置片段只会被实例化一次，即使它在多个地方被引用。
2. 依赖注入：当一个配置片段被引用时，Hydra会确保先实例化被引用的配置片段，然后再将其注入到引用它的配置片段中。
    

```py
if __name__ == "__main__":
    with initialize(version_base=None, config_path=".", config_name="config"):
        cfg = compose(config_name="config")

        # 使用Hydra和OmegaConf的工具来根据配置创建实例
        app = hydra.utils.instantiate(cfg.app)
        logger = hydra.utils.instantiate(cfg.logger)

        # 现在app和logger都使用同一个db实例
        print(app.db is logger.db)  # 输出: True
```

如果 `_target_` 放到最外层，即 `_target_` 不依靠任何节点。那么，此 YAML 配置文件以 OmegaConf 实例的方式，传递给  `@hydra.main` 修饰的 main(cfg: OmegaConf) 函数，进一步使用 `hydra.utils.get_class(cfg._target_)` 解析此配置文件对应的 class。紧接着，实例化此 class，得到对应的 Workspace 实例，再传入此 cfg 实例给它初始化和处理即可。具体例子如下：

```yaml
defaults:
  - _self_
  - task: umi
_target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace
task_name: ${task.name}
policy:
  _target_: diffusion_policy.policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy
  ...
```

对应的 diffusion_policy.workspace.train_diffusion_unet_image_workspace.py 中：

```py
OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        ...
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        ...

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
```

在 train.py 文件中，加载训练内容如下：

```py
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
```

比如，通过命令行参数 --config-name 指定 train_diffusion_unet_timm_umi_workspace，代表 cfg 为 train_diffusion_unet_timm_umi_workspace.yaml。进一步通过 hydra.utils.get_class 找到需要加载的类，从而初始化 workspace。至此，得到了对应的 `_target_` 的实例。

### 不能在配置中调用实例的方法

hydra 的 instantiate 总是会创建新的实例。由于我们的 db 没有手动创建，hydra 自动维护了这个实例，就像 Spring 的 context。但是不能在yaml中调用实例的方法，比如：

```
optimizer: # AdamW
  _target_: torch.optim.AdamW
  params: ${eval:'${model}.parameters()'} # BAD
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
```

model 此时只是 dict，并非对象，自然没有参数。但是我们可以动态插值，比如保持上面配置，在实例化时候指定传给 AdamW 参数 params 的值为 model.parameters()，具体如下：

```py
optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
```

### 命令行参数

#### 指定配置文件路径

在装饰器 `@hydra.main` 中，从指定的 config_path 目录加载 YAML 配置文件，默认加载目录下的 config.yaml 作为配置文件。如果需要加载其他配置文件，则需要指定 config-name。可以通过命令行参数 --config-name=YOUR_CONFIG 加载 YOUR_CONFIG.yaml。例如：

```bash
(umi)$ python train.py --config-name=train_diffusion_unet_timm_umi_workspace
```

### 覆盖配置文件内容

命令行传入参数优先级更高，可以覆盖 YAML 配置文件中的内容。比如：

```bash
(umi)$ python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=example_demo_session/dataset.zarr.zip
```

对于 YAML 配置中的顶层键，直接指出并覆盖。对于嵌套键，使用点 `.` 访问嵌套结构的键。即使是动态生成的键，也可以覆盖。如果键不再 YAML，命令行参数直接新增和添加到最终配置中。

`=` 覆盖列表元素，`+=` 追加到列表末尾。

```bash
python script.py data.transforms+="[normalize]"
```

覆盖字典的键，删除键，转义特殊字符：

```bash
python script.py optimizer.params='{"lr":0.01, "momentum":0.9}'
python script.py ~model.optimizer  # 删除整个 optimizer 键
python script.py "key.with.dots=value"
```

### 对 logging 的影响

logging 库初始化设置比较繁琐，而 Hydra 默认配置了 logging，并且同时输出到终端和文件。

#### 默认日志行为

当使用 `@hydra.main` 装饰器时，Hydra 会自动初始化日志系统，无需手动配置。使用如下：

```py
@hydra.main(config_path="config", config_name="default")
def main(cfg):
    logging.info("This will be captured by Hydra's logging")  # 无需手动配置 logging
```

默认日志格式： 时间戳 + 日志级别 + 模块名 + 消息（例如 `[2019-06-27 00:52:46,653][__main__][INFO] - Info level message`）

默认日志级别： INFO 级别及以上（INFO, WARNING, ERROR, CRITICAL）

每次运行时，生成唯一目录，默认在 output/<当前日期>/<时间>。日志文件路径 可通过 cfg.hydra.run.dir 在代码中访问。

## tomllib：版本 3.11 后标准库之一

tomllib 是 Python 3.11 及以上版本中新增的标准库，专门用于解析 TOML（Tom's Obvious Minimal Language）格式的配置文件。如果使用的是 Python 3.11+，可以直接使用它；若版本较低（如 Python 3.9），可以通过安装第三方库 tomli 实现相同功能。比如 OpenManus 使用了 toml 配置文件。

在 VsCode 下载 toml 语法提示的插件。比如 Even Better TOML。

例子：

```toml
# Global LLM configuration
[llm]
model = "claude-3-5-sonnet"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
max_tokens = 4096
temperature = 0.0

# Optional configuration for specific LLM models
[llm.vision]
model = "claude-3-5-sonnet"
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
```

```py
class Config:

    @staticmethod
    def _get_config_path() -> Path:
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }
        ...
```

tomllib 下的接口：
- load(fp: BinaryIO, /, *, parse_float: ParseFloat = float) -> dict[str, Any] 接收文件的 Handle
- loads(s: str, /, *, parse_float: ParseFloat = float) -> dict[str, Any] 接收字符串的配置

获取的内容，是字典形式。随后使用方括号访问，或是 get() 方法。

```py
toml_str = """
key = "value"
[table]
numbers = [1, 2, 3]
"""
data = tomllib.loads(toml_str)
print(data["table"]["numbers"])  # 输出: [1, 2, 3]
```

## Ref and Tag