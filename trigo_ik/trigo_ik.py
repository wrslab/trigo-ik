"""TrigoIK: all inverse kinematics solutions of a general 6R manipulator.

Requires only NumPy.  The chain may be given by DH parameters (standard or modified) or by a URDF
file.  Arms with more than six revolute joints are solved with the extra joints held at given
angles (``locked``).

    FK:  T = B Rz(q1) F1 Rz(q2) F2 ... Rz(q6) F6, each joint turning about the z axis of its frame.

Outline of the solver (see the paper for the derivation)
    1. q6 drops out of the third and fourth columns of  A3 A4 A5 = A2^-1 A1^-1 T F6^-1 Rz(-q6).
    2. 14 Raghavan-Roth quantities of those columns; the right side is linear in 8 products of the
       sines and cosines of (q1, q2): r = Q m + q0.  Q and q0 are exact from a 3x3 grid and a 2-D DFT.
       N, the left null space of Q, removes (q1, q2): N (left side - q0) = 0, six equations.
    3. With z = e^{iq}, the six equations and their copies times z4 form Sigma(q3) v = 0 (12x12).
       Sigma(q3) = S- e^{-iq3} + S0 + S+ e^{iq3}, exact from three values of q3 (3-point DFT).
       det Sigma = 0 is a quadratic eigenvalue problem; its unit-circle eigenvalues give q3.
    4. Per root: q4 from a 6x6 eigenvalue problem, q5 from a quadratic, (q1, q2) from the linear
       system in m, q6 from the remaining rotation.  A few Newton steps polish each solution.
Starting the elimination at each of the six joints, forwards and along the inverted loop, gives
twelve equivalent "routes".  On special geometry (intersecting or parallel axes) some routes are
degenerate.  This depends only on the geometry, so one route is selected once per robot.
"""
import math
import xml.etree.ElementTree as ET

import numpy as np

__all__ = ["TrigoIK", "SelfMotionError", "rotation", "eigvals_pencil"]


class SelfMotionError(RuntimeError):
    """all twelve routes are degenerate: the chain has continuous self-motion"""


# ---------------------------------------------------------------- transforms
def _rz(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])


def _rz_b(t):
    """Rz for an array of angles: (n,) -> (n, 4, 4)"""
    t = np.asarray(t, float)
    m = np.zeros(t.shape + (4, 4))
    c, s = np.cos(t), np.sin(t)
    m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1] = c, -s, s, c
    m[..., 2, 2] = m[..., 3, 3] = 1.0
    return m


def rotation(axis, angle):
    """4x4 rotation by angle about a unit axis through the origin (Rodrigues)"""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    M = np.eye(4)
    M[:3, :3] = np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    return M


def _translation(x=0.0, y=0.0, z=0.0):
    M = np.eye(4)
    M[:3, 3] = x, y, z
    return M


def _rpy(r, p, y):
    """URDF fixed-axis roll, pitch, yaw: Rz(y) Ry(p) Rx(r)"""
    return rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)


def _frame_on_axis(o, h):
    h = h / np.linalg.norm(h)
    x = np.cross(h, [1.0, 0.0, 0.0] if abs(h[0]) < 0.9 else [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    G = np.eye(4)
    G[:3, 0], G[:3, 1], G[:3, 2], G[:3, 3] = x, np.cross(h, x), h, o
    return G


def _wrap(q):
    return (np.asarray(q) + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------- linear algebra
_SHIFTS = (0.31 + 0.47j, -0.53 + 0.29j, 0.17 - 0.61j)              # fixed, off the unit circle


def eigvals_pencil(A, B):
    """Finite eigenvalues l of A x = l B x, NumPy only.  Shift and invert: for a shift s with A - s B
    regular, the eigenvalues m of (A - s B)^-1 B give l = s + 1/m; m = 0 marks an infinite
    eigenvalue (singular B), which is dropped."""
    for s in _SHIFTS:
        M = A - s * B
        if np.linalg.cond(M) < 1e10:
            break
    m = np.linalg.eigvals(np.linalg.solve(M, B))
    m = m[np.abs(m) > 1e-12 * max(np.abs(m).max(), 1e-300)]
    return s + 1 / m


# ---------------------------------------------------------------- batched FK, residuals, Jacobian
def _fk_batch(Q, Fs, B, frames=False):
    """FK of many configurations: Q (n, k) -> (n, 4, 4); frames: also world joint axes / origins"""
    Q = np.atleast_2d(Q)
    M = np.broadcast_to(B, (len(Q), 4, 4)).copy()
    k = Q.shape[1]
    if frames:
        W, O = np.empty((len(Q), k, 3)), np.empty((len(Q), k, 3))
    for j in range(k):
        if frames:
            W[:, j], O[:, j] = M[:, :3, 2], M[:, :3, 3]
        M = M @ _rz_b(Q[:, j]) @ Fs[j]
    return (M, W, O) if frames else M


def _pose_res_batch(Q, Fs, B, T):
    """12 pose residuals (position, rotation matrix entries) per configuration"""
    M = _fk_batch(Q, Fs, B)
    return np.concatenate([M[:, :3, 3] - T[:3, 3], (M[:, :3, :3] - T[:3, :3]).reshape(len(M), 9)], 1)


def _res_jac_batch(Q, Fs, B, T):
    """residuals (n, 12) and analytic Jacobians (n, 12, 6): joint k turns the rest of the chain about
    the world axis w_k through o_k, so dp/dq_k = w_k x (p - o_k) and dR/dq_k = [w_k]x R"""
    M, W, O = _fk_batch(Q, Fs, B, frames=True)
    p, Rm = M[:, :3, 3], M[:, :3, :3]
    r = np.concatenate([p - T[:3, 3], (Rm - T[:3, :3]).reshape(len(Q), 9)], 1)
    d = p[:, None, :] - O
    J = np.empty((len(Q), 12, 6))
    J[:, 0] = W[..., 1] * d[..., 2] - W[..., 2] * d[..., 1]
    J[:, 1] = W[..., 2] * d[..., 0] - W[..., 0] * d[..., 2]
    J[:, 2] = W[..., 0] * d[..., 1] - W[..., 1] * d[..., 0]
    Sk = np.zeros(W.shape[:2] + (3, 3))
    Sk[..., 0, 1], Sk[..., 0, 2], Sk[..., 1, 2] = -W[..., 2], W[..., 1], -W[..., 0]
    Sk[..., 1, 0], Sk[..., 2, 0], Sk[..., 2, 1] = W[..., 2], -W[..., 1], W[..., 0]
    J[:, 3:] = (Sk @ Rm[:, None]).reshape(len(Q), 6, 9).transpose(0, 2, 1)
    return r, J


def _polish_batch(Q, Fs, B, T, iters=6, tol=2e-15):
    """a few Newton steps on the 12 pose residuals, all configurations at once"""
    Q = np.array(Q, float)
    for _ in range(iters):
        r, J = _res_jac_batch(Q, Fs, B, T)
        act = np.abs(r).max(1) >= tol
        if not act.any():
            break
        dq = (np.linalg.pinv(J[act]) @ r[act][..., None])[..., 0]
        Q[act] -= dq
        if np.abs(dq).max() < 1e-15:
            break
    return Q


# ---------------------------------------------------------------- Raghavan-Roth quantities and DFT
def _q14(M):
    """the 14 Raghavan-Roth quantities of a 4x4 matrix or a stack (..., 4, 4) -> (..., 14)"""
    l, p = M[..., :3, 2], M[..., :3, 3]
    pp, pl = np.sum(p * p, -1), np.sum(p * l, -1)
    px = np.stack([p[..., 1] * l[..., 2] - p[..., 2] * l[..., 1],
                   p[..., 2] * l[..., 0] - p[..., 0] * l[..., 2],
                   p[..., 0] * l[..., 1] - p[..., 1] * l[..., 0]], -1)
    return np.concatenate([l, p, pp[..., None], pl[..., None], px, pp[..., None] * l - 2 * pl[..., None] * p], -1)


# every quantity has degree 1 in each of its two angles: a 3x3 grid and a 2-D DFT give it exactly
_G = 2 * np.pi * np.arange(3) / 3
_RZ_NEG = np.array([_rz(-a) for a in _G])


def _coeffs_from_grid(Y):
    """(3, 3, 14) values on the grid -> C (14 x 9) for the monomials
    s_a s_b, s_a c_b, c_a s_b, c_a c_b, s_a, c_a, s_b, c_b, 1"""
    c = np.fft.fft2(Y, axes=(0, 1)) / 9                          # c[k_a, k_b], k = 0, 1, -1 (index 2)
    pp, pm, p0, q0_ = c[1, 1], c[1, 2], c[1, 0], c[0, 1]
    return np.stack([-2 * (pp - pm).real, -2 * (pp + pm).imag, -2 * (pp - pm).imag, 2 * (pp + pm).real,
                     -2 * p0.imag, 2 * p0.real, -2 * q0_.imag, 2 * q0_.real, c[0, 0].real], axis=1)


def _unique(qs, tol=1e-4):
    out = []
    for q in qs:
        if all(np.linalg.norm(_wrap(q - u)) > tol for u in out):
            out.append(q)
    return out


# ---------------------------------------------------------------- one elimination route
class _Route:
    """IK of B Rz(q1) F1 ... Rz(q6) F6 = T, eliminating q3"""

    _R3 = np.random.default_rng(7).standard_normal((3, 6))       # fixed random 3x6 projection

    def __init__(self, Fs, B, T):
        self.Fs, self.B, self.T = Fs, B, T
        Ti = np.linalg.inv(B) @ T @ np.linalg.inv(Fs[5])           # Rz(q6) drops out of cols 3, 4
        F1i, F2i = np.linalg.inv(Fs[0]), np.linalg.inv(Fs[1])
        M1 = np.einsum('ij,ajk,kl->ail', F1i, _RZ_NEG, Ti)          # F1^-1 Rz(-q1) Ti   (a: q1)
        M2 = np.einsum('ij,bjk->bik', F2i, _RZ_NEG)                  # F2^-1 Rz(-q2)      (b: q2)
        Q = _coeffs_from_grid(_q14(np.einsum('bij,ajk->abik', M2, M1)))
        self.Q8, self.qc = Q[:, :8], Q[:, 8]
        U, _, _ = np.linalg.svd(self.Q8)
        self.N = U[:, 8:].T                                          # 6 x 14, N Q8 = 0
        G4 = np.array([_rz(a) @ Fs[3] for a in _G])
        G5 = np.array([_rz(b) @ Fs[4] for b in _G])
        self.G45 = np.einsum('aij,bjk->abik', G4, G5)                # (3, 3, 4, 4), independent of q3
        self.F6i = np.linalg.inv(Fs[5])
        self._S = None

    def sigma(self, t3):
        """12x12 Sigma(t3) in the basis z4^i z5^j"""
        Y = _q14(np.einsum('ij,abjk->abik', _rz(t3) @ self.Fs[2], self.G45))
        c = np.fft.fft2(Y, axes=(0, 1)) / 9
        c[0, 0] -= self.qc
        Hz = np.einsum('rk,abk->rab', self.N, c)[:, [2, 0, 1]][:, :, [2, 0, 1]]
        Sig = np.zeros((12, 4, 3), complex)
        Sig[:6, :3, :] = Hz
        Sig[6:, 1:, :] = Hz                                          # the copy times z4
        return Sig.reshape(12, 12)

    def raw_S(self):
        """S-, S0, S+ of Sigma(q3) = S- e^{-iq3} + S0 + S+ e^{iq3}, by a 3-point DFT"""
        if self._S is None:
            c = np.fft.fft(np.array([self.sigma(x) for x in _G]), axis=0) / 3
            self._S = (c[2], c[0], c[1])
        return self._S

    def usable(self, rel_tol=1e-9, seed=5):
        """rank Q = 8 and rank Sigma = 12 at three random angles; else the route loses solutions"""
        if np.linalg.matrix_rank(self.Q8, rel_tol * np.abs(self.Q8).max()) < 8:
            return False
        Sm, S0, Sp = self.raw_S()
        for t in np.random.default_rng(seed).uniform(-np.pi, np.pi, 3):
            sv = np.linalg.svd(Sm * np.exp(-1j * t) + S0 + Sp * np.exp(1j * t), compute_uv=False)
            if np.sum(sv / sv[0] > rel_tol) < 12:
                return False
        return True

    def roots_q3(self, root_tol=2e-3, same=1e-9):
        """distinct real roots q3 of det Sigma: unit-circle eigenvalues of the 24x24 linearization of
        z Sigma = S+ z^2 + S0 z + S-  (as a pencil when S+ is ill-conditioned)"""
        Sm, S0, Sp = self.raw_S()
        sc = max(np.abs(Sm).max(), np.abs(S0).max(), np.abs(Sp).max())
        A, Bm, Cm = Sp / sc, S0 / sc, Sm / sc
        I, Z = np.eye(12), np.zeros((12, 12))
        if np.linalg.cond(A) < 1e8:
            z = np.linalg.eigvals(np.block([[Z, I], [-np.linalg.solve(A, Cm), -np.linalg.solve(A, Bm)]]))
        else:
            z = eigvals_pencil(np.block([[Z, I], [-Cm, -Bm]]), np.block([[I, Z], [Z, A]]))
        th = np.sort(np.angle(z[np.isfinite(z) & (np.abs(np.abs(z) - 1) < root_tol)]))
        return th[np.r_[True, np.diff(th) > same]] if len(th) else th

    def roots_q4q5(self, t3, root_tol=2e-3, sing_tol=1e-6):
        """For all roots q3 at once.  For a known q3 the six equations read C(z4) (1, z5, z5^2)^T = 0,
        C (6x3) quadratic in z4.  q4: det(R C(z4)) = 0 with a fixed random 3x6 R, a 6x6 eigenvalue
        problem; C(z4) must then be singular.  q5: the dominant row combination of C(z4) is a quadratic
        in z5; both roots are candidates and must satisfy all six equations."""
        t3 = np.asarray(t3, float)
        m = len(t3)
        Sm, S0, Sp = self.raw_S()
        e = np.exp(1j * t3)[:, None, None]
        Hz = (Sm[:6, :9] / e + S0[:6, :9] + Sp[:6, :9] * e).reshape(m, 6, 3, 3)   # [., row, z4^a, z5^b]
        C0, C1, C2 = Hz[:, :, 0, :], Hz[:, :, 1, :], Hz[:, :, 2, :]
        A0, A1, A2 = self._R3 @ C0, self._R3 @ C1, self._R3 @ C2
        I, Z = np.eye(3), np.zeros((3, 3))
        z4 = np.full((m, 6), np.nan, complex)
        good = np.linalg.cond(A2) < 1e8
        if good.any():
            Ai = np.linalg.inv(A2[good])
            M = np.zeros((int(good.sum()), 6, 6), complex)
            M[:, :3, 3:] = I
            M[:, 3:, :3] = -Ai @ A0[good]
            M[:, 3:, 3:] = -Ai @ A1[good]
            z4[good] = np.linalg.eigvals(M)
        for i in np.where(~good)[0]:
            z = eigvals_pencil(np.block([[Z, I], [-A0[i], -A1[i]]]), np.block([[I, Z], [Z, A2[i]]]))
            z4[i, :len(z)] = z
        idx, col = np.where(np.isfinite(z4) & (np.abs(np.abs(z4) - 1) < root_tol))
        if not len(idx):
            return np.zeros(0), np.zeros(0), np.zeros(0)
        z = z4[idx, col] / np.abs(z4[idx, col])
        Cz = C0[idx] + C1[idx] * z[:, None, None] + C2[idx] * (z * z)[:, None, None]
        U, s, _ = np.linalg.svd(Cz)
        keep = s[:, -1] <= sing_tol * s[:, 0]                         # else a root of the projection only
        idx, z, Cz, U, s = idx[keep], z[keep], Cz[keep], U[keep], s[keep]
        c = np.einsum('ni,nij->nj', U[:, :, 0].conj(), Cz)            # coefficients of 1, z5, z5^2
        disc = np.sqrt(c[:, 1] ** 2 - 4 * c[:, 2] * c[:, 0])
        with np.errstate(divide='ignore', invalid='ignore'):
            W = np.stack([(-c[:, 1] + disc) / (2 * c[:, 2]), (-c[:, 1] - disc) / (2 * c[:, 2])], 1)
        rows, k = np.where(np.isfinite(W) & (np.abs(np.abs(W) - 1) < root_tol))
        w = W[rows, k] / np.abs(W[rows, k])
        y = np.stack([np.ones_like(w), w, w * w], 1)
        fine = np.linalg.norm(np.einsum('nij,nj->ni', Cz[rows], y), axis=1) <= 1e-6 * s[rows, 0] * np.sqrt(3)
        rows, w = rows[fine], w[fine]
        return t3[idx[rows]], np.angle(z[rows]), np.angle(w)

    def complete(self, t3, t4, t5):
        """(q1, q2) from the 14 equations (linear in m), q6 from the rest, Newton polish"""
        n = len(t3)
        F = self.Fs
        rhs = _q14(_rz_b(t3) @ F[2] @ _rz_b(t4) @ F[3] @ _rz_b(t5) @ F[4]) - self.qc
        m = rhs @ np.linalg.pinv(self.Q8).T
        Q = np.stack([np.arctan2(m[:, 4], m[:, 5]), np.arctan2(m[:, 6], m[:, 7]), t3, t4, t5, np.zeros(n)], 1)
        A15 = _fk_batch(Q[:, :5], F[:5], self.B)
        R6 = A15[:, :3, :3].transpose(0, 2, 1) @ (self.T @ self.F6i)[:3, :3]
        Q[:, 5] = np.arctan2(R6[:, 1, 0], R6[:, 0, 0])
        return list(_polish_batch(Q, F, self.B, self.T))

    def solve(self):
        t3, t4, t5 = self.roots_q4q5(self.roots_q3())
        return self.complete(t3, t4, t5) if len(t3) else []


def _routes(Fs, B, T):
    """the 12 equivalent write-ups of the loop, each (Fs', B', T', back), back(q') -> q"""
    I4 = np.eye(4)
    out = []
    for rev in (False, True):
        if not rev:
            G, Bg, Tg, back = list(Fs), B, T, (lambda q: q)
        else:
            Finv = [np.linalg.inv(F) for F in Fs]
            G = [Finv[4 - i] for i in range(5)] + [np.linalg.inv(B)]
            Bg, Tg, back = Finv[5], np.linalg.inv(T), (lambda q: -np.asarray(q)[::-1])
        for k in range(6):
            if k == 0:
                out.append((G, Bg, Tg, back))
                continue
            Gk = G[k:][:-1] + [G[5] @ np.linalg.inv(Tg) @ Bg] + G[:k]
            perm = list(range(k, 6)) + list(range(k))

            def back_k(q, perm=perm, back=back):
                qo = np.empty(6)
                qo[perm] = q
                return back(qo)
            out.append((Gk, I4, I4, back_k))
    order = [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11]                 # (fwd, 0) (rev, 0) (fwd, 1) ...
    return [out[i] for i in order]


# ---------------------------------------------------------------- chain descriptions
def _chain_to_BF(elems, locked):
    """elems: ('fixed', 4x4) or ('rev', axis, name), walked at zero joint angles.  A joint about axis
    h through o turns by G Rz(q) G^-1 for a frame G at o with z = h.  Returns B, Fs, joint names."""
    tf, G, names, k = np.eye(4), [], [], 0
    for e in elems:
        if e[0] == "fixed":
            tf = tf @ e[1]
            continue
        _, ax, name = e
        c = locked.get(k, locked.get(name)) if locked else None
        k += 1
        if c is not None:                                            # held joint: a constant rotation
            tf = tf @ rotation(ax, c)
        else:
            G.append(_frame_on_axis(tf[:3, 3].copy(), tf[:3, :3] @ np.asarray(ax, float)))
            names.append(name)
    if len(G) != 6:
        raise ValueError(f"TrigoIK needs exactly 6 free revolute joints, got {len(G)} "
                         "(hold the extra joints with locked={index or name: angle})")
    Fs = [np.linalg.inv(G[i]) @ G[i + 1] for i in range(5)] + [np.linalg.inv(G[5]) @ tf]
    return G[0], Fs, names


def _parse_urdf(path, base_link=None, tip_link=None):
    """joints from base_link to tip_link as ('fixed', origin) / ('rev', axis, name) elements"""
    root = ET.parse(path).getroot()
    joints = {}                                                      # child link -> joint element
    for j in root.findall("joint"):
        joints[j.find("child").get("link")] = j
    parents = {j.find("parent").get("link") for j in joints.values()}
    if base_link is None:
        roots = parents - set(joints)
        if len(roots) != 1:
            raise ValueError(f"URDF: give base_link, candidates {sorted(roots)}")
        base_link = roots.pop()
    if tip_link is None:
        leaves = set(joints) - parents
        if len(leaves) != 1:
            raise ValueError(f"URDF: give tip_link, candidates {sorted(leaves)}")
        tip_link = leaves.pop()
    path_j, link = [], tip_link
    while link != base_link:
        if link not in joints:
            raise ValueError(f"URDF: no path from {base_link} to {tip_link}")
        path_j.append(joints[link])
        link = joints[link].find("parent").get("link")
    elems = []
    for j in reversed(path_j):
        o = j.find("origin")
        xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
        rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
        elems.append(("fixed", _translation(*xyz) @ _rpy(*rpy)))
        jt = j.get("type")
        if jt in ("revolute", "continuous"):
            a = j.find("axis")
            ax = [float(v) for v in (a.get("xyz") if a is not None else "1 0 0").split()]
            elems.append(("rev", ax, j.get("name")))
        elif jt != "fixed":
            raise ValueError(f"URDF: joint {j.get('name')} of type {jt} is not supported")
    return elems


# ---------------------------------------------------------------- the solver
class TrigoIK:
    """All IK solutions of a 6R chain  FK(q) = B Rz(q1) F1 Rz(q2) F2 ... Rz(q6) F6.

    Build it with TrigoIK.from_dh(...), TrigoIK.from_urdf(...), or directly from B and F1..F6.
    The elimination route is selected once, when the solver is built."""

    def __init__(self, B, Fs, tol=1e-8, joint_names=None, seed=0):
        self.B = np.asarray(B, float)
        self.Fs = [np.asarray(F, float) for F in Fs]
        if len(self.Fs) != 6:
            raise ValueError("TrigoIK needs F1..F6")
        self.tol = tol                                               # max pose residual of a solution
        self.joint_names = joint_names
        self.route = self._select_route(seed)                        # None: continuous self-motion

    @classmethod
    def from_dh(cls, d, a, alpha, theta0=None, modified=False, locked=None, base=None, tool=None, **kw):
        """DH parameters, one entry per revolute joint.
        standard (modified=False):  A_i = Rz(q_i + theta0_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
        modified (Craig):           A_i = Rx(alpha_i) Tx(a_i) Rz(q_i + theta0_i) Tz(d_i)
        locked: {joint index (0-based): angle q_i} for joints beyond six;  base, tool: 4x4 transforms
        before the first and after the last joint."""
        d, a, alpha = (np.asarray(x, float) for x in (d, a, alpha))
        theta0 = np.zeros(len(d)) if theta0 is None else np.asarray(theta0, float)
        elems = [("fixed", np.eye(4) if base is None else np.asarray(base, float))]
        for i in range(len(d)):
            if modified:
                elems += [("fixed", rotation([1, 0, 0], alpha[i]) @ _translation(x=a[i])),
                          ("rev", [0, 0, 1], i), ("fixed", _rz(theta0[i]) @ _translation(z=d[i]))]
            else:
                elems += [("rev", [0, 0, 1], i),
                          ("fixed", _rz(theta0[i]) @ _translation(z=d[i]) @ _translation(x=a[i])
                           @ rotation([1, 0, 0], alpha[i]))]
        if tool is not None:
            elems.append(("fixed", np.asarray(tool, float)))
        B, Fs, names = _chain_to_BF(elems, locked)
        return cls(B, Fs, joint_names=names, **kw)

    @classmethod
    def from_urdf(cls, path, base_link=None, tip_link=None, locked=None, **kw):
        """URDF file; the chain from base_link to tip_link (default: the root and the single leaf).
        Fixed joints are merged; revolute and continuous joints are free unless held by
        locked={joint name or index among the revolute joints: angle}."""
        B, Fs, names = _chain_to_BF(_parse_urdf(path, base_link, tip_link), locked)
        return cls(B, Fs, joint_names=names, **kw)

    # ------------------------------------------------------------ public
    @property
    def self_motion(self):
        return self.route is None

    def fk(self, q):
        """4x4 pose of the tip for joint angles q (6,), or poses (n, 4, 4) for q (n, 6)"""
        q = np.asarray(q, float)
        M = _fk_batch(np.atleast_2d(q), self.Fs, self.B)
        return M[0] if q.ndim == 1 else M

    def ik(self, T):
        """All solutions for the tip pose T (4x4), as an (n, 6) array of angles in [-pi, pi)."""
        if self.route is None:
            raise SelfMotionError("all twelve routes are degenerate: the chain has continuous self-motion")
        T = np.array(T, dtype=float)
        U, _, Vt = np.linalg.svd(T[:3, :3])                          # re-orthonormalize the target
        T[:3, :3] = U @ Vt
        G, Bg, Tg, back = _routes(self.Fs, self.B, T)[self.route]
        cand = [back(q) for q in _Route(G, Bg, Tg).solve()]
        if not cand:
            return np.zeros((0, 6))
        ok = np.abs(_pose_res_batch(np.array(cand), self.Fs, self.B, T)).max(1) < self.tol
        sols = _unique([_wrap(q) for q, g in zip(cand, ok) if g])
        return np.array(sols).reshape(-1, 6)

    def residual(self, q, T):
        """max abs pose residual of solutions q (n, 6) for the target T"""
        return np.abs(_pose_res_batch(np.atleast_2d(q), self.Fs, self.B, np.asarray(T, float))).max(1)

    # ------------------------------------------------------------ route selection
    def _select_route(self, seed):
        """Once per robot: the first of the twelve routes with rank Q = 8 and rank Sigma = 12, tested at
        the target of a random configuration (degeneracy depends on the geometry only)."""
        q = np.random.default_rng(seed).uniform(-np.pi, np.pi, 6)
        T = self.fk(q)
        for j, (G, Bg, Tg, _) in enumerate(_routes(self.Fs, self.B, T)):
            if _Route(G, Bg, Tg).usable():
                return j
        return None
