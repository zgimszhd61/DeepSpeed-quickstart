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
