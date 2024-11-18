# MindSpeed-MM 工具库使用指南

<p align="left">
        <b>简体中文</b> |
</p>

- [Profiling采集](#jump1)
  - [静态采集](#静态采集)
  - [动态采集](#动态采集)

## <a id="jump1"></a>Profiling采集工具

套件集成了昇腾[profiling采集工具](./profiler.py)，以提供对模型运行情况的分析。内置模型均已适配，只需修改[tools.json](./tools.json)文件即可生效。

【若新增模型，请先适配如下设置】

```python
from mindspeed_mm.tools.profiler import Profiler

prof = Profiler(prof_config)
prof.start()
while train:
    train_one_step
    prof.step()
prof.stop()
```

【通用的模型config设置】

```bash
--enable                  # 指开启profiling采集
--profile_type            # 指动态或静态的profiling采集类型, static / dynamic
--ranks                   # 指profiling采集的rank, default 为-1， 指采集全部rank
```

### 静态采集

`Static Profile`静态采集功能为执行模型训练过程中的指定的steps区间进行采集, 操作步骤如下：

1. 在模型config设置里开启`enable`采集开关，设置`profile_type` 为 static, 设置 `ranks`

2. 配置静态采集的相关参数

    【静态采集的参数设置/`static_param`】

    ```bash
    --level                     # profiling采集的level选择: level0, level1, level2
    --with_stack                # 采集时是否采集算子调用栈
    --with_memory               # 采集时是否采集内存占用情况
    --record_shapes             # 采集时是否采集算子的InputShapes和InputTypes
    --with_cpu                  # 采集时是否采集CPU信息
    --save_path                 # profiling的保存路径
    --start_step                # 设置启动采集的步数
    --end_step                  # 设置结束采集的步数
    --data_simplification       # 采集时是否采用简化数据
    ```

3. 运行模型并采集profiling文件

### 动态采集

`Dynamic Profile`动态采集功能可在执行模型训练过程中随时开启采集进程，操作步骤如下：

1. 在模型config设置里开启`enable`采集开关，设置`profile_type` 为 dynamic, 设置 `ranks`

2. 配置动态采集的相关参数

    【动态采集的参数设置`dynamic_param`】

    ```bash
    --config_path               # config与log文件的路径
    ```
  
    - `config_path`指向空文件夹并自动生成`profiler_config.json`文件
    - `config_path`指已有动态配置文件`profiler_config.json`的路径

3. 运行模型

4. 在模型运行过程中，随时修改`profiler_config.json`文件配置，profiling采集会在下一个step生效并开启

    【动态采集的实现方式】

    - 动态采集通过识别`profiler_config.json`文件的状态判断文件是否被修改，若感知到`profiler_config.json`文件被修改，`dynamic_profile`会在下一个step时开启Profiling任务
    - `config_path`目录下会自动记录`dynamic_profile`的维测日志

动态采集的具体参数、入参表、及具体操作步骤等请[参考链接](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/devaids/auxiliarydevtool/atlasprofiling_16_0038.html#ZH-CN_TOPIC_0000001988052037__zh-cn_topic_0000001849812417_section17272160135118)
