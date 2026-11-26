"""Minimal use of TrigoIK: a robot from DH parameters, one from a URDF file, and a 7-axis arm with
one joint held.  Each time: random joint angles -> FK -> all IK solutions."""
import os

import numpy as np

from trigo_ik import TrigoIK

HERE = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=4, suppress=True)


def show(name, ik, rng):
    q = rng.uniform(-np.pi, np.pi, 6)
    T = ik.fk(q)
    sols = ik.ik(T)
    found = any(np.linalg.norm((s - q + np.pi) % (2 * np.pi) - np.pi) < 1e-6 for s in sols)
    print(f"{name}: {len(sols)} solutions, max pose residual {ik.residual(sols, T).max():.1e}, "
          f"generating angles among them: {found}")
    print(sols)


rng = np.random.default_rng(1)

# 1. Wei{\ss}' manipulator (standard DH, A_i = Rz(q_i) Tz(d_i) Tx(a_i) Rx(alpha_i)); up to 16 solutions
H = np.pi / 2
show("Weiss manipulator (DH)",
     TrigoIK.from_dh(d=[0, 0, 0, 0.5, 0, 0], a=[0, 0.6, 0, 0, 0.13, 0], alpha=[H, 0, -H, H, H, 0]), rng)

# 2. Universal Robots UR3e from a URDF file (base_link -> flange)
show("UR3e (URDF)", TrigoIK.from_urdf(os.path.join(HERE, "robots", "ur3e.urdf")), rng)

# 3. Franka Research 3 (7 axes) with joint 3 held at 0.4 rad: the remaining 6R chain
show("FR3, joint3 held (URDF)",
     TrigoIK.from_urdf(os.path.join(HERE, "robots", "fr3.urdf"), locked={"joint3": 0.4}), rng)
