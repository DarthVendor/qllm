"""32-direction structured local quantum search, IDEAL QUERY MODEL ONLY.

No 32-qubit circuit is executed. Candidate losses and marked fractions
are computed classically, then Grover measurement probabilities are modeled.
The Monte Carlo evaluations are excluded from oracle-query counts.
"""
import json
import numpy as np
from math import sin, asin, sqrt

Q, SAMPLES, STEPS, BUDGET = 32, 512, 12, 16

def task(seed, ntrain=120, nval=160):
    r = np.random.default_rng(seed + 400)
    vocab, dim = 4, 16
    tok = r.integers(vocab, size=(ntrain + nval, 3))
    x = np.tanh(np.eye(vocab)[tok].reshape(-1, 12) @ r.normal(0, .8, (12, dim)))
    teacher = r.normal(0, .62, (dim, vocab))
    logits = x @ teacher
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    y = np.array([r.choice(vocab, p=p) for p in probs])
    w = teacher + r.normal(0, .48, teacher.shape)
    basis = r.normal(size=(Q, dim * vocab))
    basis = np.linalg.qr(basis.T)[0].T.reshape(Q, dim, vocab)
    return x[:ntrain], y[:ntrain], x[ntrain:], y[ntrain:], w, basis

def losses(w, x, y):
    z = x @ w
    z -= z.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))

def grad(w, x, y):
    z = x @ w
    z -= z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    p[np.arange(len(y)), y] -= 1
    return x.T @ p / len(y)

def candidate_losses(z, w, basis, scale, x, y):
    weights = w[None] + scale / sqrt(Q) * np.einsum("nq,qfv->nfv", z, basis)
    a = np.einsum("tf,nfv->ntv", x, weights)
    a -= a.max(axis=-1, keepdims=True)
    logsum = np.log(np.exp(a).sum(axis=-1))
    gold = np.take_along_axis(a, np.broadcast_to(y[None, :, None], (len(z), len(y), 1)), axis=-1)[..., 0]
    return np.mean(logsum - gold, axis=1)

def adaptive_grover(r, p, budget=BUDGET):
    # BBHT-style randomized Grover depth, one verification query per shot.
    m, used = 1, 0
    angle = asin(sqrt(np.clip(p, 0, 1)))
    while used < budget:
        j = int(r.integers(m))
        cost = j + 1
        if used + cost > budget:
            j, cost = max(0, budget - used - 1), budget - used
        used += cost
        if r.random() < sin((2 * j + 1) * angle)**2:
            return True, used
        m = min(64, max(m + 1, int(np.ceil(1.34 * m))))
    return False, used

def benchmark(seeds=range(6)):
    names = ("projected_gradient", "uniform_strict_quantum", "biased_strict_quantum",
             "biased_classical", "classical_best16")
    summary = {name: [] for name in names}
    records = []
    for seed in seeds:
        x, y, xv, yv, w0, basis = task(seed)
        weights = {name: w0.copy() for name in names}
        r = np.random.default_rng(971 + seed * 11)
        counts = dict.fromkeys(names, 0)
        for step in range(STEPS):
            scale = .92 * (.08 + .92 * (1 + np.cos(np.pi * step / STEPS)) / 2)
            delta = max(.008, .065 * (1 - step / STEPS)**1.3)
            for name in names:
                w = weights[name]
                base = losses(w, x, y)
                direction = np.einsum("qfv,fv->q", basis, grad(w, x, y))
                favored = np.where(direction < 0, 1., -1.)
                if name == "projected_gradient":
                    proposal = w + scale / sqrt(Q) * np.einsum("q,qfv->fv", favored, basis)
                    if losses(proposal, x, y) < base:
                        weights[name] = proposal
                    cost, p = 1, None
                else:
                    biased = name.startswith("biased")
                    if biased:
                        prob = np.where(direction < 0, .8, .2)
                        z = np.where(r.random((SAMPLES, Q)) < prob, 1., -1.)
                    else:
                        z = r.choice((-1., 1.), size=(SAMPLES, Q))
                    cand = candidate_losses(z, w, basis, scale, x, y)
                    good = cand < base - delta
                    p = float(good.mean())
                    if name == "classical_best16":
                        inds = r.integers(SAMPLES, size=BUDGET)
                        selected = int(inds[np.argmin(cand[inds])])
                        hit, cost = cand[selected] < base, BUDGET
                    elif name == "biased_classical":
                        inds = r.integers(SAMPLES, size=BUDGET)
                        good_inds = inds[good[inds]]
                        hit, cost = len(good_inds) > 0, BUDGET
                        selected = int(good_inds[0]) if hit else 0
                    else:
                        hit, cost = adaptive_grover(r, p)
                        selected = int(r.choice(np.flatnonzero(good))) if hit and good.any() else 0
                    if hit and (name == "classical_best16" or good[selected]):
                        weights[name] = w + scale / sqrt(Q) * np.einsum("q,qfv->fv", z[selected], basis)
                counts[name] += cost
                records.append(dict(seed=seed, step=step, method=name, validation_nll=losses(weights[name], xv, yv),
                                    oracle_queries=cost, estimated_marked_fraction=p,
                                    classical_mc_candidate_evaluations=0 if name == "projected_gradient" else SAMPLES))
        for name in names:
            summary[name].append(dict(nll=losses(weights[name], xv, yv), queries=counts[name]))
    report = {name: dict(mean_validation_nll=float(np.mean([r["nll"] for r in vals])),
                         sd_validation_nll=float(np.std([r["nll"] for r in vals], ddof=1)),
                         mean_work_units=float(np.mean([r["queries"] for r in vals])))
              for name, vals in summary.items()}
    return dict(methodology="Ideal Grover probability emulation with classical candidate-loss estimation; NO 32-qubit neural oracle execution. Gradient evaluations and oracle calls are not equal cost.",
                search_qubits=Q, candidates=2**Q, results=report, records=records)

if __name__ == "__main__":
    result = benchmark()
    with open("structured_32q_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result["results"], indent=2))
