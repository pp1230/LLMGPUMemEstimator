# LLMGPUMemEstimator
The GPU RAM Estimator provides a simple tool for estimating GPU memory usage during training and inference.
## Estimator
### Usage
```python
e=Estimator()

# ----------------Default 7b model example-------------
# inference
print(f"single gpu, precision 16-bit:{e.estimate_gpu_memory_inference(precision=2)}GB")
print(f"single gpu, precision 32-bit:{e.estimate_gpu_memory_inference(precision=4)}GB")

# training
# single gpu, stage 0, memory_per_gpu(GB): 120.7826795578003, model_memory: 104.3081283569336, activation_memory: 3.43603515625, buffer_memory: 13.0385160446167
print("single gpu, stage 0, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}" \
    .format(*e.estimate_gpu_memory_training()))

# 4 gpu, stage 3, memory_per_gpu(GB): 42.5515832901001, model_memory: 26.0770320892334, activation_memory: 3.43603515625, buffer_memory: 13.0385160446167
print("4 gpu, stage 3, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}" \
    .format(*e.estimate_gpu_memory_training(stage=3, gpu_num=4)))

# 4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): 67.8596887588501, model_memory: 26.0770320892334, activation_memory: 28.744140625, buffer_memory: 13.0385160446167
print("4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}" \
    .format(*e.estimate_gpu_memory_training(stage=3, gpu_num=4, s=2048)))

# ----------------Customized model example-------------

e=Estimator(model_params=5e8, a=16, h=1024, L=24)

# inference
# single gpu, precision 16-bit:0.9313225746154785GB
print(f"single gpu, precision 16-bit:{e.estimate_gpu_memory_inference(precision=2)}GB")

# training
# single gpu, stage 0, memory_per_gpu(GB): 9.310125827789307, model_memory: 7.450580596923828, activation_memory: 0.92822265625, buffer_memory: 0.9313225746154785
print("single gpu, stage 0, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}" \
    .format(*e.estimate_gpu_memory_training()))
# 4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): 12.131858348846436, model_memory: 1.862645149230957, activation_memory: 9.337890625, buffer_memory: 0.9313225746154785
print("4 gpu, stage 3, seq_len 2048, memory_per_gpu(GB): {}, model_memory: {}, activation_memory: {}, buffer_memory: {}" \
    .format(*e.estimate_gpu_memory_training(stage=3, gpu_num=4, s=2048)))
```

## Reference
[1] Reducing Activation Recomputation in Large Transformer Models

[2] ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
