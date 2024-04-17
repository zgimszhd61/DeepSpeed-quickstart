# DeepSpeed-quickstart
 - https://github.com/microsoft/DeepSpeed



DeepSpeed是一个深度学习优化库，旨在提高大规模模型训练的速度和效率。以下是如何快速开始使用DeepSpeed框架的步骤：

### 安装DeepSpeed

1. **安装依赖项**：在安装DeepSpeed之前，需要确保已经安装了PyTorch。推荐的PyTorch版本是1.9或更高版本。此外，还需要一个CUDA或ROCm编译器，如nvcc或hipcc，用于编译C++/CUDA/HIP扩展[5]。

2. **安装DeepSpeed**：可以通过pip安装DeepSpeed，这将安装最新版本的DeepSpeed，它不依赖于特定的PyTorch或CUDA版本。可以使用以下命令安装：
   ```bash
   pip install deepspeed
   ```
   如果需要构建DeepSpeed的C++/CUDA扩展，可以使用以下命令：
   ```bash
   DS_BUILD_OPS=1 pip install deepspeed --global-option="build_ext" --global-option="-j8"
   ```
   这将加快完整构建过程的速度[4]。

### 初始化DeepSpeed

1. **配置文件**：DeepSpeed通过配置文件来管理训练参数，如批量大小、优化器、精度等。一个示例配置文件可能包含以下内容：
   ```json
   {
     "train_batch_size": 8,
     "gradient_accumulation_steps": 1,
     "optimizer": {
       "type": "Adam",
       "params": {
         "lr": 0.00015
       }
     },
     "fp16": {
       "enabled": true
     },
     "zero_optimization": true
   }
   ```
   配置文件的详细信息可以在DeepSpeed的API文档中找到[1][3]。

2. **初始化分布式环境**：如果已经设置了分布式环境，需要将`torch.distributed.init_process_group(...)`替换为`deepspeed.init_distributed()`。DeepSpeed默认使用NCCL后端，但也可以覆盖默认设置[1][3]。

3. **初始化DeepSpeed引擎**：使用`deepspeed.initialize`函数来初始化DeepSpeed引擎。这个函数会确保在底层适当地完成了分布式数据并行或混合精度训练所需的所有设置。例如：
   ```python
   model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, parameters=params, ...)
   ```
   其中`args`是命令行参数，`model`是你的模型，`params`是模型参数[1][4]。

### 训练模型

使用DeepSpeed引擎的API进行模型训练，包括前向传播、反向传播和权重更新。例如：
```python
for step, batch in enumerate(trainloader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

### 使用Azure试用DeepSpeed

DeepSpeed在Azure上的使用非常简单，推荐通过AzureML配方来试用DeepSpeed。可以在GitHub上找到作业提交和数据准备脚本[5]。

### 其他框架集成

DeepSpeed已经与多个流行的开源深度学习框架集成，例如Transformers、Lightning等。可以在这些框架的官方文档中找到如何与DeepSpeed集成的详细信息[5][7]。

这些步骤提供了一个基本的指南，用于开始使用DeepSpeed框架进行模型训练。更多详细信息和高级功能，可以参考DeepSpeed的官方文档和GitHub页面[1][2][3][4][5][6][7][8]。

Citations:
[1] https://blog.csdn.net/just_sort/article/details/131049256
[2] https://www.deepspeed.ai/getting-started/
[3] https://www.cnblogs.com/Last--Whisper/p/17939371
[4] https://cloud.tencent.com/developer/article/2314959
[5] https://github.com/microsoft/DeepSpeed
[6] https://www.deepspeed.ai/tutorials/inference-tutorial/
[7] https://docs.ray.io/en/latest/train/deepspeed.html
[8] https://github.com/microsoft/DeepSpeedExamples/blob/master/applications/DeepSpeed-Chat/README.md


-----

DeepSpeed是一个由微软开发的开源深度学习优化库，旨在提高大规模模型训练和推理的效率。虽然大部分文献和资源都集中在使用DeepSpeed进行模型训练上，但DeepSpeed也可以用于模型推理，尤其是在处理大型模型时。以下是一个使用DeepSpeed进行模型推理的快速入门指南：

### 准备工作

1. **安装DeepSpeed**：首先，需要在你的环境中安装DeepSpeed。可以通过以下命令安装：
   ```bash
   pip install deepspeed
   ```
   如果你打算使用DeepSpeed的最新特性，可以考虑从源代码安装。

2. **安装其他依赖**：根据你的模型和需求，可能还需要安装其他库，如PyTorch、Hugging Face Transformers等。

### 配置DeepSpeed

DeepSpeed推理的关键在于正确配置。DeepSpeed提供了多种优化技术，如模型并行、ZeRO优化等，这些都可以通过配置文件来启用。

1. **创建配置文件**：创建一个JSON格式的配置文件，例如`ds_config.json`，在其中指定你想要使用的DeepSpeed特性和参数。例如，启用ZeRO优化：
   ```json
   {
     "zero_optimization": {
       "stage": 2
     }
   }
   ```

2. **准备模型和数据**：确保你的模型和数据已经准备好，可以被DeepSpeed处理。对于Hugging Face的Transformers模型，你可以直接加载模型和分词器。

### 运行推理

使用DeepSpeed进行推理的具体步骤会根据你的模型和任务有所不同，但一般流程如下：

1. **加载模型**：使用DeepSpeed的API加载你的模型。如果你使用的是Hugging Face的Transformers模型，可以先加载模型，然后使用DeepSpeed的`deepspeed.initialize`方法包装模型。

2. **编写推理代码**：编写用于执行推理的代码。这通常涉及到加载数据、处理数据、使用模型进行预测以及处理预测结果。

3. **使用DeepSpeed运行推理**：使用`deepspeed`命令行工具或在你的Python脚本中使用DeepSpeed API来运行推理。确保在命令中指定配置文件和其他必要的参数。

### 示例

以下是一个简单的示例，展示了如何使用DeepSpeed进行推理：

```python
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 使用DeepSpeed初始化模型
model_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config="ds_config.json")

# 准备输入
input_text = "DeepSpeed is"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# 使用DeepSpeed模型进行推理
with torch.no_grad():
    output = model_engine(input_ids)

# 处理输出
predicted_text = tokenizer.decode(output.logits.argmax(dim=-1).squeeze().tolist(), skip_special_tokens=True)
print(predicted_text)
```

请注意，这个示例主要是为了展示如何结合使用DeepSpeed和Hugging Face的Transformers库进行推理。根据你的具体需求，可能需要调整模型加载、数据处理和推理逻辑。

### 参考资料

- DeepSpeed官方文档和GitHub仓库提供了关于安装、配置和使用DeepSpeed的详细信息[6][7]。
- Hugging Face的Transformers库文档，了解如何加载和使用预训练模型[2]。

这个快速入门指南提供了使用DeepSpeed进行模型推理的基本步骤和示例代码。根据你的具体模型和任务，可能需要进一步调整和优化。

Citations:
[1] http://giantpandacv.com/project/PyTorch/%E3%80%90DeepSpeed%20%E6%95%99%E7%A8%8B%E7%BF%BB%E8%AF%91%E3%80%91%E5%BC%80%E5%A7%8B%EF%BC%8C%E5%AE%89%E8%A3%85%E7%BB%86%E8%8A%82%E5%92%8CCIFAR-10%20Tutorial/
[2] https://cloud.tencent.com/developer/article/2314984
[3] https://blog.csdn.net/weixin_43301333/article/details/127237122
[4] http://www.rhkb.cn/news/118151.html
[5] https://wqw547243068.github.io/dist
[6] https://docs.aws.amazon.com/zh_cn/sagemaker/latest/dg/large-model-inference-tutorials-deepspeed-djl.html
[7] https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-chat/chinese/README.md
[8] https://www.bilibili.com/read/mobile?id=21515099

DeepSpeed是一个由微软开发的开源深度学习优化库，专门用于加速大规模模型的训练。它通过一系列优化技术，如ZeRO（零冗余优化器），提高了训练速度，降低了成本，并提高了可用性。以下是使用DeepSpeed进行预训练的快速入门指南。

## 安装DeepSpeed

首先，您需要安装DeepSpeed。这可以通过Python的包管理器pip来完成。在您的终端或命令提示符中运行以下命令：

```bash
pip install deepspeed
```

如果您在Windows环境下，可能需要设置环境变量以屏蔽async_io库：

```bash
set DS_BUILD_AIO=0
set DS_BUILD_SPARSE_ATTN=0
```

然后，您可以继续安装：

```bash
python setup.py bdist_wheel
```

## 配置DeepSpeed

DeepSpeed的配置是通过JSON文件进行的。您需要创建一个名为`ds_config.json`的文件，并在其中指定训练参数，例如批处理大小、梯度累积步骤、优化器类型等。以下是一个配置文件的示例：

```json
{
  "train_batch_size": 8,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "Adam",
    "params": {
      "lr": 0.00015
    }
  },
  "fp16": {
    "enabled": true
  },
  "zero_optimization": true
}
```

## 初始化DeepSpeed

在您的训练脚本中，您需要使用DeepSpeed提供的API来初始化模型和优化器。这通常是通过调用`deepspeed.initialize`方法来完成的。以下是一个初始化DeepSpeed引擎的示例：

```python
import deepspeed

model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)
```

这里的`cmd_args`是一个解析命令行参数的对象，`model`是您的PyTorch模型，`params`是模型参数的列表。

## 训练模型

一旦DeepSpeed引擎初始化完成，您可以使用它来进行前向传播、反向传播和权重更新。以下是一个训练循环的示例：

```python
for step, batch in enumerate(data_loader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

在这里，`data_loader`是您的数据加载器，`model_engine`是DeepSpeed创建的包装器，它封装了您的模型。

## 多GPU和多节点训练

DeepSpeed支持单机多GPU和多节点训练。您可以使用`deepspeed`命令行工具来启动分布式训练。例如，以下命令使用8个GPU在2个节点上运行训练：

```bash
deepspeed --num_gpus 8 --num_nodes 2 --hostfile hostfile --master_addr hostname1 --master_port=9901 your_program.py
```

在这里，`hostfile`是一个包含节点地址的文件，`your_program.py`是您的训练脚本。

## 资源

- DeepSpeed官方文档提供了详细的安装和配置指南[1][5]。
- CSDN博客和其他技术博客提供了关于如何使用DeepSpeed的实战经验和教程[2][3][4][7][8]。
- Hugging Face的文档中也介绍了如何将DeepSpeed集成到其Trainer API中[6]。

请注意，这只是一个快速入门指南，具体细节可能会根据您的具体需求和环境而有所不同。更多详细信息和高级配置选项，请参考DeepSpeed的官方文档和相关资源。

Citations:
[1] https://note.iawen.com/note/llm/deepspeed
[2] https://blog.csdn.net/weixin_42486623/article/details/132761712
[3] https://www.cnblogs.com/Last--Whisper/p/17939371
[4] https://cloud.tencent.com/developer/article/2314984
[5] http://giantpandacv.com/project/PyTorch/%E3%80%90DeepSpeed%20%E6%95%99%E7%A8%8B%E7%BF%BB%E8%AF%91%E3%80%91%E5%BC%80%E5%A7%8B%EF%BC%8C%E5%AE%89%E8%A3%85%E7%BB%86%E8%8A%82%E5%92%8CCIFAR-10%20Tutorial/
[6] https://huggingface.co/docs/transformers/v4.36.1/zh/main_classes/deepspeed
[7] https://github.com/Tencent/TencentPretrain/wiki/DeepSpeed%E6%94%AF%E6%8C%81
[8] https://blog.csdn.net/qq_29707567/article/details/132495789
