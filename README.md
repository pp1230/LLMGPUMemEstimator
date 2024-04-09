# LLMGPUMemEstimator
The GPU RAM Estimator provides a simple tool for estimating GPU memory occupation during training and inference. This tool is suitable for GPT-type models based on transformers.
## Estimator
### Params define
a number of attention heads 

p pipeline parallel size 

b microbatch sizes sequence length 

h hidden dimension size

t tensor parallel size 

L number of transformer layers 

v vocabulary size
### Usage
```python
from Estimator import Estimator

e=Estimator()

# ----------------Default 7b model example-------------
# inference
'''
single gpu, precision 16-bit:13.04GB
single gpu, precision 32-bit:26.08GB
'''
print("single gpu, precision 16-bit:{:.2f}GB".format(e.estimate_gpu_memory_inference(precision=2)))
print("single gpu, precision 32-bit:{:.2f}GB".format(e.estimate_gpu_memory_inference(precision=4)))

# training
'''
single gpu, stage 0, memory_per_gpu(GB):          120.78, model_memory: 104.31, activation_memory: 3.44, buffer_memory: 13.04
4 gpu, stage 3, memory_per_gpu(GB):               42.55, model_memory: 26.08, activation_memory: 3.44, buffer_memory: 13.04
4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): 67.86, model_memory: 26.08, activation_memory: 28.74, buffer_memory: 13.04
'''
print("single gpu, stage 0, memory_per_gpu(GB): {:.2f}, model_memory: {:.2f}, activation_memory: {:.2f}, buffer_memory: {:.2f}" \
    .format(*e.estimate_gpu_memory_training()))
print("4 gpu, stage 3, memory_per_gpu(GB): {:.2f}, model_memory: {:.2f}, activation_memory: {:.2f}, buffer_memory: {:.2f}" \
    .format(*e.estimate_gpu_memory_training(stage=3, gpu_num=4)))
print("4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): {:.2f}, model_memory: {:.2f}, activation_memory: {:.2f}, buffer_memory: {:.2f}" \
    .format(*e.estimate_gpu_memory_training(stage=3, gpu_num=4, s=2048)))

# ----------------Customized model example-------------

e=Estimator(model_params=5e8, a=16, h=1024, L=24)

# inference
# single gpu, precision 16-bit:0.93GB
print("single gpu, precision 16-bit:{:.2f}GB".format(e.estimate_gpu_memory_inference(precision=2)))

# training
'''
single gpu, stage 0, memory_per_gpu(GB):          9.31, model_memory: 7.45, activation_memory: 0.93, buffer_memory: 0.93
4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): 12.13, model_memory: 1.86, activation_memory: 9.34, buffer_memory: 0.93
'''
print("single gpu, stage 0, memory_per_gpu(GB): {:.2f}, model_memory: {:.2f}, activation_memory: {:.2f}, buffer_memory: {:.2f}" \
    .format(*e.estimate_gpu_memory_training()))
print("4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): {:.2f}, model_memory: {:.2f}, activation_memory: {:.2f}, buffer_memory: {:.2f}" \
    .format(*e.estimate_gpu_memory_training(stage=3, gpu_num=4, s=2048)))
```

## Reference
[1] Reducing Activation Recomputation in Large Transformer Models

[2] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
