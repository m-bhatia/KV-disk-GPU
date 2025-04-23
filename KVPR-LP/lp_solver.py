import cvxpy as cp

# Model/system parameters (example values)
batch_size = 32             # batch size
sequence_len = 1024         # total sequence length
embedding_dim = 4096        # embedding dimension
precision = 2               # bytes per element (e.g., FP16)

# Hardware characteristics (example values)
v_gpu = 312e12   # FLOPs/sec (A100 FP16 peak ~312 TFLOPS)
v_com = 32e9     # PCIe bandwidth in bytes/sec

# Optimization variable
l = cp.Variable(integer=True)

# Memory and computation expressions
M_X = batch_size * l * embedding_dim * precision
M_KV_remain = 2 * batch_size * (sequence_len - l) * embedding_dim * precision
N_KV_recomp = 4 * batch_size * l * embedding_dim**2

# Time expressions
t_activation = M_X / v_com
t_recomp = N_KV_recomp / v_gpu
t_KV_transfer = M_KV_remain / v_com

# Total time
t_total = t_activation + cp.maximum(t_recomp, t_KV_transfer)

# Constraints
constraints = [
    l >= 0,
    l <= s
]

# Define and solve the problem
prob = cp.Problem(cp.Minimize(t_total), constraints)
prob.solve()

print("Optimal split point l* =", round(l.value))
print("Minimum total time (s) =", t_total.value)
