#!/home/bingxing2/ailab/scxlab0004/.conda/envs/llama_factory/bin/python
import socket

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('', 0))
addr = s.getsockname()
print(addr[1])
s.close()