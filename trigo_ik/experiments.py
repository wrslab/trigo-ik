"""Smaller-scale versions of the experiments of the paper.  NumPy only.

    python experiments.py              # all
    python experiments.py random weiss # a selection: random, weiss, arms6, arms7

Timings depend on the machine and the NumPy / BLAS build; they show the trend only.
"""
import os
import sys
import time

import numpy as np

import trigo_ik as ti
from trigo_ik import TrigoIK

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = lambda name: os.path.join(HERE, "robots", name + ".urdf")
H = np.pi / 2

# DH tables (standard DH): d, a, alpha
WEISS = dict(d=[0, 0, 0, 0.5, 0, 0], a=[0, 0.6, 0, 0, 0.13, 0], alpha=[H, 0, -H, H, H, 0])
IRB140 = dict(d=[0.352, 0, 0, 0.380, 0, 0.065], a=[0.070, 0.360, 0, 0, 0, 0], alpha=[-H, 0, -H, H, -H, 0])
XARM7 = dict(d=[0.267, 0, 0.293, 0, 0.3425, 0, 0.097], a=[0, 0, 0.0525, 0.0775, 0, 0.076, 0],
             alpha=[-H, H, H, H, H, -H, 0])
IIWA14 = dict(d=[0.360, 0, 0.420, 0, 0.400, 0, 0.126], a=[0] * 7, alpha=[-H, H, H, -H, -H, H, 0])


# ---------------------------------------------------------------- helpers
def wrap(q):
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi


def contains(sols, q, tol=1e-4):
    return any(np.linalg.norm(wrap(s - q)) < tol for s in sols)


def timed_ik(ik, T, reps=3):
    """solutions and the best of reps runs, ms"""
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        sols = ik.ik(T)
        best = min(best, time.perf_counter() - t0)
    return sols, best * 1e3


def numeric_all(ik, T, n_starts, rng, iters=60, tol=1e-9):
    """Reference: Levenberg-Marquardt on the 12 pose residuals from n_starts random starts (all at
    once), distinct converged solutions.  Independent of the elimination."""
    Q = rng.uniform(-np.pi, np.pi, (n_starts, 6))
    lam = np.full(n_starts, 1e-2)
    for _ in range(iters):
        r, J = ti._res_jac_batch(Q, ik.Fs, ik.B, T)
        JT = J.transpose(0, 2, 1)
        dq = np.linalg.solve(JT @ J + lam[:, None, None] * np.eye(6), (JT @ r[..., None]))[..., 0]
        Qn = Q - dq
        better = (ti._pose_res_batch(Qn, ik.Fs, ik.B, T) ** 2).sum(1) < (r ** 2).sum(1)
        Q[better] = Qn[better]
        lam = np.where(better, lam / 3, lam * 3)
    ok = ik.residual(Q, T) < tol
    return ti._unique([wrap(q) for q in Q[ok]])


def missed_extra(ref, sols):
    return sum(not contains(sols, q) for q in ref), sum(not contains(ref, s) for s in sols)


def usable_routes(ik):
    T = ik.fk(np.random.default_rng(0).uniform(-np.pi, np.pi, 6))
    return sum(ti._Route(G, Bg, Tg).usable() for G, Bg, Tg, _ in ti._routes(ik.Fs, ik.B, T))


def hist(counts):
    k, v = np.unique(counts, return_counts=True)
    return " ".join(f"{a}:{b}" for a, b in zip(k, v))


def report(name, counts, times, res, missed, extra=""):
    print(f"  {name:36s} solutions {hist(counts):34s} time median {np.median(times):5.2f} ms, "
          f"p99 {np.percentile(times, 99):5.2f} ms   max residual {max(res):.0e}   "
          f"generating angles missed {missed}{extra}")


# ---------------------------------------------------------------- experiments
def exp_random(n=200, n_check=10):
    """random 6R chains: d, a in [-0.6, 0.6] m, alpha in [-pi, pi); one random target each"""
    print("Random 6R chains")
    rng = np.random.default_rng(0)
    counts, times, res, missed, chk = [], [], [], 0, []
    for i in range(n):
        ik = TrigoIK.from_dh(d=rng.uniform(-0.6, 0.6, 6), a=rng.uniform(-0.6, 0.6, 6),
                             alpha=rng.uniform(-np.pi, np.pi, 6))
        q = rng.uniform(-np.pi, np.pi, 6)
        T = ik.fk(q)
        sols, dt = timed_ik(ik, T)
        counts.append(len(sols)); times.append(dt); res.append(ik.residual(sols, T).max())
        missed += not contains(sols, q)
        if i < n_check:
            chk.append(missed_extra(numeric_all(ik, T, 2000, rng), sols))
    agree = sum(c == (0, 0) for c in chk)
    report(f"{n} chains", counts, times, res, missed, f"   numerical reference agrees {agree}/{len(chk)}")


def exp_weiss(n=1000, n_check=3):
    """Wei{\\ss}' manipulator, which reaches 16 solutions"""
    print("Weiss manipulator (16 solutions)")
    rng = np.random.default_rng(9)
    ik = TrigoIK.from_dh(**WEISS)
    counts, times, res, missed, chk = [], [], [], 0, []
    for i in range(n):
        q = rng.uniform(-np.pi, np.pi, 6)
        T = ik.fk(q)
        sols, dt = timed_ik(ik, T, reps=1)
        counts.append(len(sols)); times.append(dt); res.append(ik.residual(sols, T).max())
        missed += not contains(sols, q)
        if len(sols) == 16 and len(chk) < n_check:
            chk.append(missed_extra(numeric_all(ik, T, 3000, rng), sols))
    agree = sum(c == (0, 0) for c in chk)
    report(f"{n} targets, {usable_routes(ik)}/12 routes", counts, times, res, missed,
           f"   numerical reference agrees on 16-solution targets {agree}/{len(chk)}")


def exp_arms6(n=300):
    """six-axis commercial arms"""
    print("Six-axis commercial arms")
    arms = [("ABB IRB 140 (DH)", lambda: TrigoIK.from_dh(**IRB140))] + \
        [(f"{lab} (URDF)", lambda f=f: TrigoIK.from_urdf(URDF(f)))
         for lab, f in (("UR3e", "ur3e"), ("Kawasaki RS007L", "rs007l"), ("DENSO COBOTTA", "cobotta"),
                        ("UFACTORY Lite 6", "lite6"), ("FANUC CRX-5iA", "crx5ia"))]
    for name, make in arms:
        rng = np.random.default_rng(6)
        ik = make()
        counts, times, res, missed = [], [], [], 0
        for _ in range(n):
            q = rng.uniform(-np.pi, np.pi, 6)
            T = ik.fk(q)
            sols, dt = timed_ik(ik, T)
            counts.append(len(sols)); times.append(dt); res.append(ik.residual(sols, T).max())
            missed += not contains(sols, q)
        report(f"{name}, {usable_routes(ik)}/12 routes", counts, times, res, missed)


def exp_arms7(n=30):
    """seven-axis arms: each joint in turn is held at a random angle per target"""
    print("Seven-axis arms, one joint held")
    arms = [("Franka Research 3 (URDF)", lambda L: TrigoIK.from_urdf(URDF("fr3"), locked=L)),
            ("UFACTORY xArm 7 (DH)", lambda L: TrigoIK.from_dh(**XARM7, locked=L)),
            ("KUKA LBR iiwa 14 (DH)", lambda L: TrigoIK.from_dh(**IIWA14, locked=L)),
            ("Tianji Marvin M6-S (URDF)", lambda L: TrigoIK.from_urdf(URDF("marvin_m6s"), locked=L))]
    for name, make in arms:
        print(f" {name}")
        for k in range(7):
            rng = np.random.default_rng(6)
            counts, times, res, missed, routes = [], [], [], 0, []
            for _ in range(n):
                ik = make({k: rng.uniform(-np.pi, np.pi)})           # route selected here, untimed
                if ik.self_motion:
                    break
                routes.append(usable_routes(ik))
                q = rng.uniform(-np.pi, np.pi, 6)
                T = ik.fk(q)
                sols, dt = timed_ik(ik, T)
                counts.append(len(sols)); times.append(dt); res.append(ik.residual(sols, T).max())
                missed += not contains(sols, q)
            if not counts:
                print(f"  q{k + 1} held{'':28s} all twelve routes degenerate: continuous self-motion")
            else:
                report(f"q{k + 1} held, {min(routes)}-{max(routes)}/12 routes", counts, times, res, missed)


if __name__ == "__main__":
    ALL = {"random": exp_random, "weiss": exp_weiss, "arms6": exp_arms6, "arms7": exp_arms7}
    print(f"NumPy {np.__version__}")
    for key in sys.argv[1:] or list(ALL):
        ALL[key]()
