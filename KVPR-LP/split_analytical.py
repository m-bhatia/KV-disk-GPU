# Model/system parameters (example values)
batch_size = 32             # batch size
sequence_len = 1024         # total sequence length
embedding_dim = 4096        # embedding dimension
precision = 2               # bytes per element (e.g., FP16)

# Hardware characteristics (example values)
v_gpu = 312e12   # FLOPs/sec (A100 FP16 peak ~312 TFLOPS)
v_com = 32e9     # PCIe bandwidth in bytes/sec

# calculate optimal recomputation/load split
l = (sequence_len * precision * v_gpu) / (2 * embedding_dim * v_com + precision * v_gpu)
print("Optimal split point l* =", round(l))
