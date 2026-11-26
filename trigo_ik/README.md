# TrigoIK

Source code for the paper *A Fast and Complete Inverse Kinematics Solver for General 6R Manipulators*.

TrigoIK returns all inverse kinematics solutions of a manipulator with six revolute joints, up to 16.
It assumes no special structure, such as a spherical wrist or parallel axes. Robots with more than
six revolute joints are solved with the extra joints held at given angles.

**Requirements:** Python 3.8 or newer and NumPy. Only the optional drawing script `show_solutions.py` also needs matplotlib.

## Files

| File | Contents |
|---|---|
| `trigo_ik.py` | The solver, in a single file. It reads DH parameters or a URDF file. |
| `example.py` | Minimal use: a DH robot, a URDF robot, and a 7-axis arm with one joint held. |
| `experiments.py` | Smaller-scale versions of the experiments in Section VI of the paper. |
| `show_solutions.py` | Draws all solutions of a target, one 3-D view each. This is the only file that needs matplotlib. |
| `LICENSE` | MIT License. |
| `robots/*.urdf` | Kinematics-only URDF files (joint frames, axes and limits, no meshes) of the commercial arms used in the experiments. |

## Usage

```python
import numpy as np
from trigo_ik import TrigoIK

# 1. From DH parameters (standard DH: A_i = Rz(q_i + theta0_i) Tz(d_i) Tx(a_i) Rx(alpha_i))
H = np.pi / 2
ik = TrigoIK.from_dh(d=[0, 0, 0, 0.5, 0, 0], a=[0, 0.6, 0, 0, 0.13, 0], alpha=[H, 0, -H, H, H, 0])

# 2. From a URDF file (the chain from base_link to tip_link)
ik = TrigoIK.from_urdf("robots/ur3e.urdf", base_link="base_link", tip_link="flange")

# 3. A 7-axis arm with joint 3 held at 0.4 rad
ik = TrigoIK.from_urdf("robots/fr3.urdf", locked={"joint3": 0.4})

T = ik.fk([0.1, -0.5, 0.3, -1.2, 0.4, 0.9])   # 4x4 pose of the tip
Q = ik.ik(T)                                   # (n, 6) array, all solutions, angles in [-pi, pi)
print(len(Q), ik.residual(Q, T).max())
```

**DH input**
- `TrigoIK.from_dh(d, a, alpha, theta0=None, modified=False, locked=None, base=None, tool=None)` takes one entry per joint.
- `modified=True` selects the modified (Craig) convention, A_i = Rx(alpha_i) Tx(a_i) Rz(q_i + theta0_i) Tz(d_i).
- `base` and `tool` are optional 4x4 transforms before the first and after the last joint.

**URDF input**
- `TrigoIK.from_urdf(path, base_link=None, tip_link=None, locked=None)` reads the chain from `base_link` to `tip_link`. By default these are the root link and the single leaf link.
- Fixed joints are merged into their neighbors.
- Revolute and continuous joints are free unless they are held. Prismatic joints are not supported.

**Holding joints**
- `locked={joint: angle}` holds a joint at the given angle.
- The joint is given by its index among the revolute joints (0-based) or by its URDF joint name.
- Exactly six joints must remain free.

**Direct input**
- `TrigoIK(B, Fs)` accepts the chain as constant 4x4 transforms, FK(q) = B Rz(q1) F1 Rz(q2) F2 ... Rz(q6) F6, where each joint turns about the z axis of its frame.

**Joint limits** are not applied. `ik` returns every solution with its angles wrapped to [-pi, pi), and you can filter or shift them by 2π against your own limits.

**Continuous self-motion:** if all twelve elimination routes are degenerate, the chain has continuous self-motion and `ik.self_motion` is `True`. For example, the KUKA LBR iiwa with its elbow joint held is such a chain. In that case `ik.ik(T)` raises `SelfMotionError` instead of returning an empty set.

## Drawing the solutions

```
python show_solutions.py                      # FANUC CRX-5iA, the target of random joint angles
python show_solutions.py robots/ur3e.urdf     # another URDF robot
python show_solutions.py out.png              # save to a file instead of opening a window
```

From your own code, call `show_solutions.show(ik, T)`.

Each solution is drawn as a skeleton:
- every joint is a short thick bar from its center along the positive axis direction;
- every link is a straight line between consecutive joint centers;
- each joint has the same color as the link it moves;
- the fixed base and the tool point are black.

Two solutions can lie on the same axis lines and differ only in the axis directions, as in Weiß' manipulator. The one-sided bars make such pairs distinguishable.

## Method in brief

1. q6 drops out of two columns of the loop equation.
2. Fourteen Raghavan–Roth quantities are linear in eight products of sines and cosines of (q1, q2). Their coefficients are obtained exactly from a 3 × 3 grid of values by a 2-D DFT, and a left null space removes (q1, q2).
3. The remaining six equations and their copies form the 12 × 12 matrix Σ(q3) = S₋e^{−iq3} + S₀ + S₊e^{iq3}, whose coefficients come from a 3-point DFT. The unit-circle eigenvalues of its 24 × 24 linearization give q3.
4. For each root, q4 comes from a 6 × 6 eigenvalue problem, q5 from a quadratic, (q1, q2) from a linear system, and q6 from the remaining rotation. A few Newton steps polish each solution.
5. Starting at each of the six joints, forwards and along the inverted loop, gives twelve equivalent routes. The first non-degenerate route is selected once, when the solver is built.

When the leading coefficient S₊ is ill-conditioned, the quadratic eigenvalue problem is solved as a matrix pencil by shift and invert (`eigvals_pencil`), so no QZ routine is needed.

## Experiments

Run all of them with `python experiments.py`, or a selection with `python experiments.py random weiss arms6 arms7`. Together they take about half a minute.

| Name | Paper | Contents |
|---|---|---|
| `random` | VI-A | 200 random 6R chains, cross-checked against a multi-start Levenberg–Marquardt reference |
| `weiss` | VI-A | Weiß' manipulator, which reaches 16 solutions |
| `arms6` | VI-B | Six-axis commercial arms: the ABB IRB 140 from DH parameters and five arms from URDF files |
| `arms7` | VI-C | Seven-axis arms with each joint held in turn: the FR3 and Marvin M6-S from URDF files, and the xArm 7 and LBR iiwa 14 from DH parameters |

Each line reports:
- the histogram of solution counts;
- the median and 99th-percentile time;
- the largest pose residual;
- whether the generating joint angles were missed, which should be 0 everywhere.

The experiments are smaller than in the paper. Timings depend on the machine and the NumPy/BLAS build, so they show the trend only.
