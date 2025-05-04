# formulas derived from KVPR paper

"""
s: sequence length
h: embedding dimension
p: precision
v_gpu: GPU processing speed
v_com: data transmission speed
"""
def compute_analytical_split(s, h, p, v_gpu, v_com):
    return (1/1.013) * (s * p * v_gpu) / (2 * h * v_com + p * v_gpu)

def compute_new_analytical_split(s, h, p, v_gpu, v_com):
    return ((1/1.026) * s * (p * v_gpu - h * v_com)) / (2 * h * v_com + p * v_gpu)

def main():
    seq_lengths = [2 ** i for i in range(7, 21)]
    splits = []
    for s in seq_lengths:
        l = compute_new_analytical_split(s, 4096, 2, 312e12, 32e9)
        # print(f"seq length: {s}\tsplit point l*: {l}")
        splits.append(l)
    # print("Optimal split point l* =", round(l))
    print(splits)

if __name__ == "__main__":
    main()
