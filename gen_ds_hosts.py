#!/home/bingxing2/ailab/scxlab0004/.conda/envs/llama_factory/bin/python
# 导入必要的模块
import os
import socket
import subprocess

# 定义一个函数，用于从slurm环境变量中获取节点列表
def get_nodelist():
    # 获取slurm分配的节点列表，例如node[01-04]
    nodelist = os.environ.get("SLURM_JOB_NODELIST")
    # 如果没有获取到节点列表，返回空列表
    if not nodelist:
        return []
    # 使用scontrol命令将节点列表展开为单个节点，例如node01,node02,node03,node04
    # 参考[1](https://stackoverflow.com/questions/67908731/how-to-run-a-python-script-through-slurm-in-a-cluster)
    cmd = ["scontrol", "show", "hostnames", nodelist]
    output = subprocess.check_output(cmd)
    # 将输出转换为字符串，并按换行符分割为列表
    nodes = output.decode().split("\n")
    # 去除列表中的空元素
    nodes = [node for node in nodes if node]
    # 返回节点列表
    return nodes

# 定义一个函数，用于根据节点列表生成hosts文件
def generate_hosts(nodes):
    # 定义一个空字符串，用于存储hosts文件的内容
    hosts = ""
    try:
        gpus_per_node = os.environ.get("CUDA_VISIBLE_DEVICES").split(",").__len__()
    except:
        gpus_per_node = 0
    # 遍历节点列表
    for node in nodes:
        # 获取节点的IP地址，参考[2](http://homeowmorphism.com/2017/04/18/Python-Slurm-Cluster-Five-Minutes)
        ip = socket.gethostbyname(node)
        # 将节点的IP地址和主机名拼接为一行，参考[3](https://stackoverflow.com/questions/65439640/hostfile-with-mpirun-on-multinode-with-slurm)
        line = node +f" slots={gpus_per_node}" + "\n"
        # 将该行追加到hosts文件的内容中
        hosts += line
    # 返回hosts文件的内容
    return hosts

# 定义一个函数，用于将hosts文件的内容写入到指定的文件路径
def write_hosts(hosts, filepath):
    # 以写入模式打开文件，参考[4](https://www.hpc.caltech.edu/documentation/slurm-commands)
    with open(filepath, "w") as f:
        # 将hosts文件的内容写入到文件中
        f.write(hosts)

# 调用上述函数，生成并写入hosts文件
# 获取节点列表
nodes = get_nodelist()
# 生成hosts文件的内容
hosts = generate_hosts(nodes)
# 定义要写入的文件路径，可以根据需要修改
filepath = f"{os.environ.get('SACC_HOME')}/configs/{os.environ.get('SACC_UUID')}/ds_hosts"
# 将hosts文件的内容写入到文件路径
write_hosts(hosts, filepath)
# 打印成功信息
print("Hosts file generated and written to " + filepath)
