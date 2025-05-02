# formulas derived from KVPR paper

"""
s: sequence length
h: embedding dimension
p: precision
v_gpu: GPU processing speed
v_com: data transmission speed
"""
def compute_analytical_split(s, h, p, v_gpu, v_com):
    return (s * p * v_gpu) / (2 * h * v_com + p * v_gpu)

def main():
    l = compute_analytical_split(1024, 4096, 2, 312e12, 32e9)
    print("Optimal split point l* =", round(l))

if __name__ == "__main__":
    main()
