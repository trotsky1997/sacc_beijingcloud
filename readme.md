# SACC使用指南

SACC是AI化学组推出的全网最好用的🐛❤🤗❤🚀 Slurm+Huggingface+Deepspeed大模型训练工具，因为我们实在找不到类似工具，所以也是全网最好用的😀

## 安装

```Bash
git clone https://github.com/trotsky1997/sacc_beijingcloud.git
cd sacc
bash ./install.sh
```

## 使用

```Bash
 sacc  --num_nodes 2 --gpu_per_nodes 4 --cpu_per_nodes 16 --mem_per_cpu 8  [目标训练脚本及其参数]
```

### 示例

```Bash
cd LLaMA-Factory/
sacc  --num_nodes 2 --gpu_per_nodes 4 --cpu_per_nodes 16 --mem_per_cpu 8   src/train_bash.py  --stage pt     --model_name_or_path microsoft/phi-1_5    --do_train     --dataset chemllm     --finetuning_type full    --output_dir phi-1_5_checkpoint_2     --overwrite_cache     --lr_scheduler_type cosine     --logging_steps 10     --save_steps 1000     --learning_rate 5e-5     --num_train_epochs 3.0     --plot_loss     --bf16 --overwrite_output_dir --streaming --max_steps 9999999999
```

## 参数

- --num_nodes：这个选项指定了您的作业的节点数。一个节点是一个运行您的代码的单个计算机或服务器。默认值是 2，这意味着您的作业将在两个节点上运行。您可以通过传递一个不同的整数来改变这个值，例如 --num_nodes 4。
- --gpu_per_nodes：这个选项指定了每个节点的 GPU 数量。GPU 是图形处理单元，可以通过执行并行计算来加速您的代码。默认值是 8，这意味着每个节点将有 8 个 GPU 可用。您可以通过传递一个不同的整数来改变这个值，例如 --gpu_per_nodes 2。
- --cpu_per_nodes：这个选项指定了每个节点的 CPU 数量。CPU 是中央处理单元，顺序执行您的代码。默认值是 8，这意味着每个节点将有 8 个 CPU 可用。您可以通过传递一个不同的整数来改变这个值，例如 --cpu_per_nodes 16。
- --mem_per_cpu：这个选项指定了每个 CPU 的内存数量。内存是在您的代码运行时存储您的数据和变量的空间。默认值是 8，这意味着每个 CPU 将有 8 GB 的内存可用。您可以通过传递一个不同的整数来改变这个值，例如 --mem_per_cpu 4。
- --job_name：这个选项指定了您的作业的名称。作业名称是一个字符串，用于标识您的作业并帮助您跟踪它。默认值是您的用户名后面加上 "_no_name"，这意味着您的作业名称将是类似于 "alice_no_name" 的东西。您可以通过传递一个不同的字符串来改变这个值，例如 --job_name "my_awesome_job"。
- --partition：这个选项指定了您的作业将运行在哪个集群分区。分区是一组具有相似特征和可用性的节点。默认值是 "AI4Phys"，这意味着您的作业将运行在 AI4Phys 分区，这是专门用于人工智能和物理研究的分区。您可以通过传递一个不同的字符串来改变这个值，例如 --partition "general"。

## Parameters

- --num_nodes: This option specifies the number of nodes for your job. A node is a single computer or server that runs your code. The default value is 2, which means your job will run on two nodes. You can change this value by passing a different integer to the option, such as --num_nodes 4.
- --gpu_per_nodes: This option specifies the number of GPUs per node for your job. A GPU is a graphics processing unit that can accelerate your code by performing parallel computations. The default value is 8, which means each node will have 8 GPUs available. You can change this value by passing a different integer to the option, such as --gpu_per_nodes 2.
- --cpu_per_nodes: This option specifies the number of CPUs per node for your job. A CPU is a central processing unit that executes your code sequentially. The default value is 8, which means each node will have 8 CPUs available. You can change this value by passing a different integer to the option, such as --cpu_per_nodes 16.
- --mem_per_cpu: This option specifies the number of GBs of memory per CPU for your job. Memory is the storage space that holds your data and variables while your code is running. The default value is 8, which means each CPU will have 8 GBs of memory available. You can change this value by passing a different integer to the option, such as --mem_per_cpu 4.
- --job_name: This option specifies the name of your job. A job name is a string that identifies your job and helps you keep track of it. The default value is your username followed by "_no_name", which means your job name will be something like "alice_no_name". You can change this value by passing a different string to the option, such as --job_name "my_awesome_job".
- --partition: This option specifies the partition of the cluster that your job will run on. A partition is a group of nodes that have similar characteristics and availability. The default value is "AI4Phys", which means your job will run on the AI4Phys partition, which is dedicated to artificial intelligence and physics research. You can change this value by passing a different string to the option, such as --partition "general".
