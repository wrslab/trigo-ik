"""Draw all IK solutions of a target, one 3-D view per solution.  Needs matplotlib (the solver does not).

    python show_solutions.py                      # FANUC CRX-5iA, the target of random joint angles
    python show_solutions.py robots/ur3e.urdf     # another URDF robot
    python show_solutions.py out.png              # save instead of opening a window

Each robot is drawn as a skeleton: every joint as a short thick bar from its center along the positive
axis direction (two solutions may differ only in that direction), every link as a straight line
between consecutive joint centers, each joint in the same color as the link it moves.
The fixed base and the tool point are black.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import trigo_ik as ti
from trigo_ik import TrigoIK

JNT_RGB = [(0, .45, .70), (.90, .50, .05), (0, .62, .45), (.80, .25, .20), (.55, .35, .65), (.35, .70, .90)]
INK = (0.1, 0.1, 0.1)


def skeleton(ik, q):
    """joint axes W (6, 3), joint centers O (6, 3), base origin, tool point"""
    M, W, O = ti._fk_batch(np.asarray(q, float)[None], ik.Fs, ik.B, frames=True)
    return W[0], O[0], np.zeros(3), M[0, :3, 3]


def draw(ax, ik, q, L):
    W, O, base, tip = skeleton(ik, q)
    pts = [base] + list(O) + [tip]
    for k in range(len(pts) - 1):                                     # links: base -> J1 -> ... -> tool
        c = INK if k == 0 else JNT_RGB[k - 1]
        ax.plot(*np.array([pts[k], pts[k + 1]]).T, color=c, lw=3)
    for k in range(6):                                                # joints: bars along +axis
        ax.plot(*np.array([O[k], O[k] + W[k] * L]).T, color=JNT_RGB[k], lw=10,
                solid_capstyle="butt")
    ax.scatter(*base, color=INK, s=40, marker="s")
    ax.scatter(*tip, color=INK, s=25)


def show(ik, T, sols=None, file=None, ncols=4):
    """one 3-D view per solution of the target T, all views at the same scale"""
    sols = ik.ik(T) if sols is None else sols
    n = len(sols)
    if n == 0:
        print("no solution")
        return
    P = np.vstack([np.vstack([O, base, tip]) for _, O, base, tip in (skeleton(ik, q) for q in sols)])
    ctr, half = (P.max(0) + P.min(0)) / 2, (P.max(0) - P.min(0)).max() / 2 * 1.05
    L = 0.25 * half
    cols = min(n, ncols)
    rows = -(-n // cols)
    fig = plt.figure(figsize=(2.6 * cols, 2.6 * rows + 0.6))
    for i, q in enumerate(sols):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        draw(ax, ik, q, L)
        for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), ctr):
            set_lim(c - half, c + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.1)
        ax.set_axis_off()
        ax.set_title(f"Solution {i + 1}", fontsize=10)
    fig.suptitle(f"{n} solutions, max pose residual {ik.residual(sols, T).max():.0e}")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - 0.6 / (2.6 * rows + 0.6), wspace=0, hspace=0.1)
    if file:
        fig.savefig(file, dpi=150)
    else:
        plt.show()


if __name__ == "__main__":
    args = sys.argv[1:]
    urdf = args.pop(0) if args and args[0].endswith(".urdf") else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "robots", "crx5ia.urdf")
    ik = TrigoIK.from_urdf(urdf)
    T = ik.fk(np.random.default_rng(0).uniform(-np.pi, np.pi, 6))
    show(ik, T, file=args[0] if args else None)
