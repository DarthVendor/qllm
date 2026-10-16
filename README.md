# QLLM: Quantum-Guided Training for Classical Autoregressive Models

Experimental research into quantum amplitude amplification for selecting classical neural-network weight updates. Selected weights can be mirrored directly into an ordinary autoregressive predictor.

> **Research prototype, not a demonstrated quantum speedup.** Some experiments use actual gate-level reversible arithmetic, while the larger 12-, 16-, and 32-bit studies simulate *ideal quantum oracle-query probabilities* from classically estimated candidate losses. No full 32-qubit transformer loss oracle or hardware timing result has been demonstrated.

## Research goals

- Train bias-free autoregressive models using quantum-guided discrete or low-rank parameter updates.
- Compare Grover/adaptive search and cosine-decayed update scales with AdamW, SPSA, random and gradient-informed classical search.
- Implement compute–mark–uncompute reversible neural-loss oracles and count logical qubits/gates.
- Evaluate validation negative log-likelihood and query/gate resource costs fairly.

## Key experimental findings

| Experiment | Result | Major caveat |
|---|---|---|
| Small reversible Boolean oracle | Grover amplification selects correct weight strings | Simplified parity/nonlinear Boolean tasks |
| Reversible arithmetic v4 | 15 logical qubits, integer-logit prediction and zero-one error marking | Not a reversible cross-entropy circuit |
| Adaptive decay, 6-D adapter | Quantum ~1.17417 vs AdamW ~1.17383 validation NLL | Ideal query simulation; classical loss computation |
| 12/16 binary weights | Quantum 1.872 / 0.662 vs continuous AdamW 2.690 / 0.350 | Discrete vs continuous and very small validation sets |
| 32-D selective search | Quantum gradient-biased 0.939, classical biased 0.936, projected gradient 0.932 | Six small tasks; 32-qubit ideal-query **emulation** |

The complete scripts, original JSON data, and plots were assembled into a downloadable archive in the ChatGPT conversation. This repo begins with the documented experiments; consult the original scripts before citing any quantitative result.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Scripts are independent experiment programs rather than a unified library.

## Methodological warnings

1. **Oracle-query savings are not training-time savings.** Reversibly computing a full neural loss and uncomputing scratch registers can be expensive.
2. A 32-qubit search *register* does not imply the full loss circuit uses only 32 qubits.
3. Comparisons between binary quantum parameters and continuous AdamW parameters are not capacity-matched.
4. Grover over-rotation explains why the probability of marked configurations can decline after the optimum number of iterations.
5. Entanglement or quantum superposition does not allow reading out exponentially many trained parameter vectors.
6. Early 12-bit variance was inflated by only two held-out examples per seed and by mixing dataset difficulty with optimizer randomness.

## Next steps

- Gate-level reversible quantized cross-entropy or polynomial surrogate loss
- Tests of gate counts/depth, ancillas, and coherent threshold-check cost
- Paired datasets with more validation samples, uncertainty intervals, matched-precision classical baselines
- Scaling studies in 16/32/64-dimensional *local* weight update spaces

## License

MIT.
