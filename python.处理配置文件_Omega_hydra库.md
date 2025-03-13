---
id: qb5cah4vrkew5znsyd9u9ur
title: 处理配置文件_Omega_hydra库
desc: ''
updated: 1741877635867
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

如果 `_target_` 放到最外层，即 `_target_` 不依靠任何节点。那么，此 YAML 配置文件以 OmegaConf 实例的方式，传递给 `_target_` 指定的 class，或者是 `@hydra.main` 修饰的 main(cfg: OmegaConf) 函数，后续可通过此参数访问所有配置内容。

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

### 在命令行传入参数，覆盖配置文件内容

命令行传入的参数优先级更高，可以覆盖 YAML 配置文件中的内容。比如：

```bash
(umi)$ python train.py --config-name=train_diffusion_unet_timm_umi_workspace \
    task.dataset_path=example_demo_session/dataset.zarr.zip
```


hydra 的 instantiate 总是会创建新的实例。由于我们的 db 没有手动创建，hydra 自动维护了这个实例，就像 Spring 的 context。

但是注意，不能在yaml中调用实例的方法，比如：

```
optimizer: # AdamW
  _target_: torch.optim.AdamW
  params: ${eval:'${model}.parameters()'}
  lr: 1.0e-4
  weight_decay: 0.01
  betas: [0.9, 0.95]
```

model 此时只是 dict，并非对象，自然没有参数。但是我们可以动态插值，比如保持上面配置，在实例化时候指定传给 AdamW 参数 params 的值为 model.parameters()，具体如下：

```py
optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
```

hydra的instantiate
如果配置中定义_target_，那么会根据指定类实例化对象。如果你的应用程序有更复杂的配置结构，例如嵌套的配置组，可以继续使用 _target_ 来指定嵌套的类或函数。instantiate 方法会根据配置文件中的 _target_ 键动态地实例化相应的类，并使用配置文件中的参数来构造对象。Hydra 会递归地解析配置组中的所有_target_键，并实例化相应的类。

hydra对logging会有影响
使用logging的输出，最终会被放置到目录output下，并且以时间命名，随后再${task_name}.log下找到。
https://hydra.cc/docs/tutorials/basic/running_your_app/logging/
https://hydra.cc/docs/configure_hydra/logging/
import logging
from omegaconf import DictConfig
import hydra

# A logger for this file
log = logging.getLogger(__name__)

@hydra.main()
def my_app(_cfg: DictConfig) -> None:
    log.info("Info level message")
    log.debug("Debug level message")

if __name__ == "__main__":
    my_app()

$ python my_app.py
[2019-06-27 00:52:46,653][__main__][INFO] - Info level message

设置输出格式
日志输出会保存，设置如下：


formatters:
  simple:
    format: '[%(asctime)s][%(name)s][%(levelname)s] - %(message)s'
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: simple
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: INFO
    formatter: simple
    # filename: ${hydra.job.name}.log
    filename: ./log/${logging.name}.log
loggers:
  my_logger:
    level: DEBUG
    handlers: [console, file]
    propagate: false
  hydra:
    handlers: [console]
    level: INFO
    propagate: false
root:
  level: DEBUG
  handlers: [console, file]

最后使用logging.getLogger获取注册的日志，比如my_logger
import logging
from omegaconf import DictConfig
import hydra

@hydra.main(config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # 获取日志记录器
    logger = logging.getLogger('my_logger')
    
    # 使用日志记录器记录信息
    logger.info('Hello from Hydra!')
    
if __name__ == "__main__":
    main()

最后输出文件会保存在如outputs/2024-07-08/13-38-33/diffusion_train.log的位置。


## Ref and Tag