# LLMGPUMemEstimator
The GPU RAM Estimator provides a simple tool for estimating GPU memory usage during training and inference.
## Estimator
### Usage
```python
e=Estimator()

# single gpu, stage 0, memory_per_gpu(GB): 120.7826795578003, model_memory: 104.3081283569336, activation_memory: 3.43603515625, buffer_memory: 13.0385160446167
print("single gpu, stage 0, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}".format(*e.estimator_gpu_memory()))

# 4 gpu, stage 3, memory_per_gpu(GB): 42.5515832901001, model_memory: 26.0770320892334, activation_memory: 3.43603515625, buffer_memory: 13.0385160446167
print("4 gpu, stage 3, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}".format(*e.estimator_gpu_memory(stage=3, gpu_num=4)))

# 4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): 67.8596887588501, model_memory: 26.0770320892334, activation_memory: 28.744140625, buffer_memory: 13.0385160446167
print("4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}".format(*e.estimator_gpu_memory(stage=3, gpu_num=4, s=2048)))
```

## Reference
[1] Reducing Activation Recomputation in Large Transformer Models
[2] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
