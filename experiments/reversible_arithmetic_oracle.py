"""Gate-level reversible integer-logit autoregressive predictor.
4 binary weights, 3 training examples, 15 logical qubits.
Statevector implements X/CNOT/multi-controlled-X, H, phase marking and uncomputation.
No candidate-loss lookup drives the oracle. Uses classification error, NOT cross entropy.
"""
import json
import numpy as np
from pathlib import Path

NQ = 15
WEIGHTS = tuple(range(4))
SCORES = ((4, 5), (6, 7))
EQ, T1, T2, GT = 8, 9, 10, 11
ERRORS = (12, 13, 14)
SIZE = 1 << NQ
INDEX = np.arange(SIZE, dtype=np.uint32)

def xgate(state, target, controls=(), zero_controls=()):
    mask = ((INDEX >> target) & 1) == 0
    for c in controls:
        mask &= ((INDEX >> c) & 1) == 1
    for c in zero_controls:
        mask &= ((INDEX >> c) & 1) == 0
    a = INDEX[mask]
    b = a ^ (1 << target)
    state[a], state[b] = state[b].copy(), state[a].copy()

def hadamard(state, q):
    a = INDEX[((INDEX >> q) & 1) == 0]
    b = a ^ (1 << q)
    u, v = state[a].copy(), state[b].copy()
    state[a] = (u + v) / np.sqrt(2)
    state[b] = (u - v) / np.sqrt(2)

def apply(state, ops):
    for target, controls, zeros in ops:
        xgate(state, target, controls, zeros)

def example_ops(context, label, error_q):
    ops = []
    def gate(target, controls=(), zeros=()):
        ops.append((target, tuple(controls), tuple(zeros)))
    for cl, score in enumerate(SCORES):
        for feature in range(2):
            if context[feature]:
                w = 2 * cl + feature
                gate(score[1], (w, score[0]))
                gate(score[0], (w,))
    # Unsigned 2-bit comparator: predict 1 iff score_1 > score_0.
    gate(EQ, (SCORES[0][1],))
    gate(EQ, (SCORES[1][1],))
    gate(EQ)
    gate(T1, (SCORES[1][1],), (SCORES[0][1],))
    gate(T2, (EQ, SCORES[1][0]), (SCORES[0][0],))
    gate(GT, (T1,))
    gate(GT, (T2,))
    gate(error_q, (GT,))
    if label:
        gate(error_q)
    temporary = ops[:-2] if label else ops[:-1]
    ops.extend(reversed(temporary))
    return ops

def make_oracle(samples):
    ops = []
    for (context, label), error_q in zip(samples, ERRORS):
        ops.extend(example_ops(context, label, error_q))
    return ops

def oracle(state, ops):
    apply(state, ops)
    zero_errors = np.ones(SIZE, dtype=bool)
    for q in ERRORS:
        zero_errors &= ((INDEX >> q) & 1) == 0
    state[zero_errors] *= -1
    apply(state, ops[::-1])

def diffusion(state):
    for q in WEIGHTS:
        hadamard(state, q)
        xgate(state, q)
    all_ones = np.ones(SIZE, dtype=bool)
    for q in WEIGHTS:
        all_ones &= ((INDEX >> q) & 1) == 1
    state[all_ones] *= -1
    for q in WEIGHTS:
        xgate(state, q)
        hadamard(state, q)

def predict(bits, context):
    score0 = sum(bits[j] * context[j] for j in range(2))
    score1 = sum(bits[j + 2] * context[j] for j in range(2))
    return int(score1 > score0)

def run(samples, iterations):
    state = np.zeros(SIZE, dtype=np.complex128)
    state[0] = 1
    for q in WEIGHTS:
        hadamard(state, q)
    ops = make_oracle(samples)
    for _ in range(iterations):
        oracle(state, ops)
        diffusion(state)
    return np.abs(state[:16]) ** 2, ops

def main():
    contexts = [(0, 1), (1, 0), (1, 1)]
    teachers = [(1, 0, 0, 1), (0, 1, 1, 0), (1, 1, 0, 0), (1, 0, 1, 1)]
    results = []
    for teacher in teachers:
        samples = [(context, predict(teacher, context)) for context in contexts]
        good = [i for i in range(16) if all(
            predict([(i >> j) & 1 for j in range(4)], context) == label
            for context, label in samples)]
        ops = make_oracle(samples)
        for i in range(16):
            state = np.zeros(SIZE, dtype=np.complex128)
            state[i] = 1
            oracle(state, ops)
            assert np.count_nonzero(abs(state) > 1e-10) == 1
            assert np.isclose(state[i], -1 if i in good else 1)
        successes = []
        for k in range(5):
            prob, _ = run(samples, k)
            observed = float(sum(prob[i] for i in good))
            angle = np.arcsin(np.sqrt(len(good) / 16))
            expected = float(np.sin((2 * k + 1) * angle)**2)
            assert abs(observed - expected) < 1e-9
            successes.append(observed)
        results.append(dict(teacher=teacher, valid_configs=len(good),
                            success_probability=successes,
                            x_family_operations_per_oracle=2*len(ops)))
    Path("results").mkdir(exist_ok=True)
    Path("results/reversible_arithmetic_results.json").write_text(
        json.dumps(dict(logical_qubits=NQ, task="integer logits and zero-one error",
                        caveat="No reversible cross entropy; ideal multi-controlled gates", runs=results), indent=2))
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
