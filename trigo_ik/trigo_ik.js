// TrigoIK: all inverse kinematics solutions of a general 6R manipulator.  JavaScript port of
// trigo_ik.py, with no dependencies (ES module, browser or Node).
//
//   FK:  T = B Rz(q1) F1 Rz(q2) F2 ... Rz(q6) F6, each joint turning about the z axis of its frame.
//
//   const ik = TrigoIK.fromDH({ d, a, alpha });          // or TrigoIK.fromURDF(text), new TrigoIK(B, Fs)
//   const T = ik.fk([0.1, -0.5, 0.3, -1.2, 0.4, 0.9]);   // 4x4 as nested arrays
//   const Q = ik.ik(T);                                   // all solutions, angles in [-pi, pi)
//
// The algorithm is the same as in trigo_ik.py; see there and the paper for the derivation.  The
// linear algebra that NumPy takes from LAPACK is written out here: complex eigenvalues by Hessenberg
// reduction and shifted QR, singular values by one-sided Jacobi, linear systems by LU.

const TAU = 2 * Math.PI;
const G3 = [0, TAU / 3, 2 * TAU / 3];
const W3 = [0, 1, 2].map(k => [0, 1, 2].map(a => [Math.cos(-TAU * k * a / 3), Math.sin(-TAU * k * a / 3)]));
const SHIFTS = [[0.31, 0.47], [-0.53, 0.29], [0.17, -0.61]];       // fixed, off the unit circle
const R3 = [                                                        // fixed random 3x6 projection
  [0.0012301533574825742, 0.2987455375084699, -0.2741378553622176, -0.8905918387572742, -0.45467078517172255, -0.9916465549964624],
  [0.060143602597438485, 1.3402152455545335, -0.49220651855132963, -0.6204748998199404, 0.4898420501851982, 0.35688700816006075],
  [0.10541424899789856, -0.9304680447082047, -0.02925182246327349, 0.6953031944582878, -1.344214547285082, -0.45761576104021817]];
const SELECT_Q = [0.8605556614246863, -1.4464727375963786, -2.8841484100105235, -3.03774645687452, 1.9683349641197863, 2.5934197786078093];
const SELECT_T = [1.9163898891237885, 1.9348490455536158, 0.0962933399642707];

export class SelfMotionError extends Error {
  constructor() { super('all twelve routes are degenerate: the chain has continuous self-motion'); this.name = 'SelfMotionError'; }
}

// ---------------------------------------------------------------- 4x4 transforms (Float64Array(16), row-major)
function eye4() { const M = new Float64Array(16); M[0] = M[5] = M[10] = M[15] = 1; return M; }
function mul4(A, B) {
  const C = new Float64Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) {
    C[4 * i + j] = A[4 * i] * B[j] + A[4 * i + 1] * B[4 + j] + A[4 * i + 2] * B[8 + j] + A[4 * i + 3] * B[12 + j];
  }
  return C;
}
function mul4s(...Ms) { return Ms.reduce((A, B) => mul4(A, B)); }
function inv4(A) {                                                  // general 4x4 inverse, Gauss-Jordan
  const M = Array.from({ length: 4 }, (_, i) => [A[4 * i], A[4 * i + 1], A[4 * i + 2], A[4 * i + 3], +(i === 0), +(i === 1), +(i === 2), +(i === 3)]);
  for (let c = 0; c < 4; c++) {
    let p = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c];
    for (let j = 0; j < 8; j++) M[c][j] /= d;
    for (let r = 0; r < 4; r++) if (r !== c) {
      const f = M[r][c];
      if (f) for (let j = 0; j < 8; j++) M[r][j] -= f * M[c][j];
    }
  }
  const C = new Float64Array(16);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) C[4 * i + j] = M[i][4 + j];
  return C;
}
function rz(t) { const M = eye4(), c = Math.cos(t), s = Math.sin(t); M[0] = c; M[1] = -s; M[4] = s; M[5] = c; return M; }
export function rotation(axis, angle) {
  const n = Math.hypot(axis[0], axis[1], axis[2]), k = axis.map(v => v / n);
  const c = Math.cos(angle), s = Math.sin(angle), v = 1 - c, M = eye4();
  M[0] = c + k[0] * k[0] * v;        M[1] = k[0] * k[1] * v - k[2] * s; M[2] = k[0] * k[2] * v + k[1] * s;
  M[4] = k[1] * k[0] * v + k[2] * s; M[5] = c + k[1] * k[1] * v;        M[6] = k[1] * k[2] * v - k[0] * s;
  M[8] = k[2] * k[0] * v - k[1] * s; M[9] = k[2] * k[1] * v + k[0] * s; M[10] = c + k[2] * k[2] * v;
  return M;
}
function translation(x = 0, y = 0, z = 0) { const M = eye4(); M[3] = x; M[7] = y; M[11] = z; return M; }
function rpy(r, p, y) { return mul4s(rotation([0, 0, 1], y), rotation([0, 1, 0], p), rotation([1, 0, 0], r)); }
function frameOnAxis(o, h) {
  const hn = Math.hypot(...h); h = h.map(v => v / hn);
  const e = Math.abs(h[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  let x = cross(h, e); const xn = Math.hypot(...x); x = x.map(v => v / xn);
  const y = cross(h, x), G = eye4();
  for (let i = 0; i < 3; i++) { G[4 * i] = x[i]; G[4 * i + 1] = y[i]; G[4 * i + 2] = h[i]; G[4 * i + 3] = o[i]; }
  return G;
}
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function toM4(T) { return T.length === 16 ? Float64Array.from(T) : Float64Array.from(T.flat()); }
function toNested(M) { return [0, 1, 2, 3].map(i => Array.from(M.slice(4 * i, 4 * i + 4))); }
const wrap = q => ((q + Math.PI) % TAU + TAU) % TAU - Math.PI;

function fk(q, Fs, B) {
  let M = B;
  for (let j = 0; j < q.length; j++) M = mul4(mul4(M, rz(q[j])), Fs[j]);
  return M;
}
function poseRes(q, Fs, B, T) {
  const M = fk(q, Fs, B), r = new Float64Array(12);
  for (let i = 0; i < 3; i++) {
    r[i] = M[4 * i + 3] - T[4 * i + 3];
    for (let j = 0; j < 3; j++) r[3 + 3 * i + j] = M[4 * i + j] - T[4 * i + j];
  }
  return r;
}
const maxAbs = r => r.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

// ---------------------------------------------------------------- complex dense matrices {m, n, re, im}
function cmat(m, n) { return { m, n, re: new Float64Array(m * n), im: new Float64Array(m * n) }; }
function ceye(n) { const A = cmat(n, n); for (let i = 0; i < n; i++) A.re[i * n + i] = 1; return A; }
function cfromReal(m, n, data) { const A = cmat(m, n); A.re.set(data); return A; }
function cmul(A, B) {
  const C = cmat(A.m, B.n);
  for (let i = 0; i < A.m; i++) for (let k = 0; k < A.n; k++) {
    const ar = A.re[i * A.n + k], ai = A.im[i * A.n + k];
    if (ar === 0 && ai === 0) continue;
    for (let j = 0; j < B.n; j++) {
      const br = B.re[k * B.n + j], bi = B.im[k * B.n + j];
      C.re[i * C.n + j] += ar * br - ai * bi;
      C.im[i * C.n + j] += ar * bi + ai * br;
    }
  }
  return C;
}
function cabsMax(A) { let m = 0; for (let i = 0; i < A.re.length; i++) m = Math.max(m, Math.hypot(A.re[i], A.im[i])); return m; }
function cscale(A, sr, si = 0) {
  const C = cmat(A.m, A.n);
  for (let i = 0; i < A.re.length; i++) { C.re[i] = A.re[i] * sr - A.im[i] * si; C.im[i] = A.re[i] * si + A.im[i] * sr; }
  return C;
}
function caxpy(A, B, sr, si) {                                     // A + (sr + i si) B
  const C = cmat(A.m, A.n);
  for (let i = 0; i < A.re.length; i++) {
    C.re[i] = A.re[i] + B.re[i] * sr - B.im[i] * si; C.im[i] = A.im[i] + B.re[i] * si + B.im[i] * sr;
  }
  return C;
}
function cblock(blocks) {                                          // 2x2 grid of equal square blocks
  const k = blocks[0][0].n, C = cmat(2 * k, 2 * k);
  for (let bi = 0; bi < 2; bi++) for (let bj = 0; bj < 2; bj++) {
    const X = blocks[bi][bj];
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      C.re[(bi * k + i) * 2 * k + bj * k + j] = X.re[i * k + j];
      C.im[(bi * k + i) * 2 * k + bj * k + j] = X.im[i * k + j];
    }
  }
  return C;
}

function csolve(A, B) {                                            // A X = B, LU with partial pivoting
  const n = A.n, m = B.n, a = { re: A.re.slice(), im: A.im.slice() }, x = { re: B.re.slice(), im: B.im.slice() };
  for (let c = 0; c < n; c++) {
    let p = c, best = -1;
    for (let r = c; r < n; r++) { const v = Math.hypot(a.re[r * n + c], a.im[r * n + c]); if (v > best) { best = v; p = r; } }
    if (p !== c) {
      for (let j = 0; j < n; j++) swap(a, c * n + j, p * n + j);
      for (let j = 0; j < m; j++) swap(x, c * m + j, p * m + j);
    }
    const dr = a.re[c * n + c], di = a.im[c * n + c], dd = dr * dr + di * di;
    for (let r = c + 1; r < n; r++) {
      const nr = a.re[r * n + c], ni = a.im[r * n + c];
      if (nr === 0 && ni === 0) continue;
      const fr = (nr * dr + ni * di) / dd, fi = (ni * dr - nr * di) / dd;
      for (let j = c; j < n; j++) {
        const ur = a.re[c * n + j], ui = a.im[c * n + j];
        a.re[r * n + j] -= fr * ur - fi * ui; a.im[r * n + j] -= fr * ui + fi * ur;
      }
      for (let j = 0; j < m; j++) {
        const ur = x.re[c * m + j], ui = x.im[c * m + j];
        x.re[r * m + j] -= fr * ur - fi * ui; x.im[r * m + j] -= fr * ui + fi * ur;
      }
    }
  }
  for (let c = n - 1; c >= 0; c--) {
    const dr = a.re[c * n + c], di = a.im[c * n + c], dd = dr * dr + di * di;
    for (let j = 0; j < m; j++) {
      let sr = x.re[c * m + j], si = x.im[c * m + j];
      for (let k = c + 1; k < n; k++) {
        const ar = a.re[c * n + k], ai = a.im[c * n + k], xr = x.re[k * m + j], xi = x.im[k * m + j];
        sr -= ar * xr - ai * xi; si -= ar * xi + ai * xr;
      }
      x.re[c * m + j] = (sr * dr + si * di) / dd; x.im[c * m + j] = (si * dr - sr * di) / dd;
    }
  }
  return { m: n, n: m, re: x.re, im: x.im };
}
function swap(a, i, j) { let t = a.re[i]; a.re[i] = a.re[j]; a.re[j] = t; t = a.im[i]; a.im[i] = a.im[j]; a.im[j] = t; }

function csvd(A) {
  // one-sided Jacobi on the columns (m >= n): A V = U diag(s); returns s (descending), U (m x n), V (n x n)
  const m = A.m, n = A.n, a = { re: A.re.slice(), im: A.im.slice() }, V = ceye(n);
  for (let sweep = 0; sweep < 60; sweep++) {
    let rotated = false;
    for (let p = 0; p < n - 1; p++) for (let q = p + 1; q < n; q++) {
      let al = 0, be = 0, gr = 0, gi = 0;
      for (let i = 0; i < m; i++) {
        const pr = a.re[i * n + p], pi = a.im[i * n + p], qr = a.re[i * n + q], qi = a.im[i * n + q];
        al += pr * pr + pi * pi; be += qr * qr + qi * qi;
        gr += pr * qr + pi * qi; gi += pr * qi - pi * qr;               // conj(a_p) . a_q
      }
      const g = Math.hypot(gr, gi);
      if (g === 0 || g <= 1e-15 * Math.sqrt(al * be)) continue;
      rotated = true;
      const er = gr / g, ei = -gi / g;                                  // e^{-i phi}
      const zeta = (be - al) / (2 * g);
      const t = (zeta >= 0 ? 1 : -1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
      const c = 1 / Math.sqrt(1 + t * t), s = c * t;
      for (const [X, rows, cols] of [[a, m, n], [V, n, n]]) {
        for (let i = 0; i < rows; i++) {
          const ip = i * cols + p, iq = i * cols + q;
          const qr = X.re[iq] * er - X.im[iq] * ei, qi = X.re[iq] * ei + X.im[iq] * er;   // a_q e^{-i phi}
          const pr = X.re[ip], pi = X.im[ip];
          X.re[ip] = c * pr - s * qr; X.im[ip] = c * pi - s * qi;
          X.re[iq] = s * pr + c * qr; X.im[iq] = s * pi + c * qi;
        }
      }
    }
    if (!rotated) break;
  }
  const sv = [];
  for (let j = 0; j < n; j++) {
    let t = 0;
    for (let i = 0; i < m; i++) t += a.re[i * n + j] ** 2 + a.im[i * n + j] ** 2;
    sv.push([Math.sqrt(t), j]);
  }
  sv.sort((x, y) => y[0] - x[0]);
  const s = sv.map(v => v[0]), U = cmat(m, n), Vo = cmat(n, n);
  sv.forEach(([sj, j], k) => {
    for (let i = 0; i < m; i++) {
      U.re[i * n + k] = sj > 0 ? a.re[i * n + j] / sj : 0; U.im[i * n + k] = sj > 0 ? a.im[i * n + j] / sj : 0;
    }
    for (let i = 0; i < n; i++) { Vo.re[i * n + k] = V.re[i * n + j]; Vo.im[i * n + k] = V.im[i * n + j]; }
  });
  return { s, U, V: Vo };
}
function ccond(A) { const s = csvd(A).s; return s[s.length - 1] > 0 ? s[0] / s[s.length - 1] : Infinity; }
function cpinv(A) {                                                // m >= n
  const { s, U, V } = csvd(A), n = A.n, m = A.m, P = cmat(n, m), cut = 1e-15 * s[0];
  for (let k = 0; k < n; k++) {
    if (s[k] <= cut) continue;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const vr = V.re[i * n + k], vi = V.im[i * n + k], ur = U.re[j * n + k], ui = -U.im[j * n + k];
      P.re[i * m + j] += (vr * ur - vi * ui) / s[k]; P.im[i * m + j] += (vr * ui + vi * ur) / s[k];
    }
  }
  return P;
}

function ceigvals(A) {
  // eigenvalues of a general complex matrix: Householder reduction to Hessenberg form, then
  // single-shift QR with Wilkinson shifts and deflation.  Returns [[re, im], ...].
  const n = A.n, hr = A.re.slice(), hi = A.im.slice(), idx = (i, j) => i * n + j;
  for (let k = 0; k < n - 2; k++) {
    let nx = 0;
    for (let i = k + 1; i < n; i++) nx += hr[idx(i, k)] ** 2 + hi[idx(i, k)] ** 2;
    nx = Math.sqrt(nx);
    if (nx === 0) continue;
    const x0r = hr[idx(k + 1, k)], x0i = hi[idx(k + 1, k)], x0 = Math.hypot(x0r, x0i);
    const ph = x0 > 0 ? [x0r / x0, x0i / x0] : [1, 0];
    const vr = new Float64Array(n), vi = new Float64Array(n);
    for (let i = k + 1; i < n; i++) { vr[i] = hr[idx(i, k)]; vi[i] = hi[idx(i, k)]; }
    vr[k + 1] += ph[0] * nx; vi[k + 1] += ph[1] * nx;               // v = x + e^{i arg x0} |x| e1
    let vn = 0;
    for (let i = k + 1; i < n; i++) vn += vr[i] ** 2 + vi[i] ** 2;
    if (vn === 0) continue;
    for (let j = 0; j < n; j++) {                                     // H = (I - 2 v v^H / v^H v) H
      let sr = 0, si = 0;
      for (let i = k + 1; i < n; i++) { sr += vr[i] * hr[idx(i, j)] + vi[i] * hi[idx(i, j)]; si += vr[i] * hi[idx(i, j)] - vi[i] * hr[idx(i, j)]; }
      sr *= 2 / vn; si *= 2 / vn;
      for (let i = k + 1; i < n; i++) { hr[idx(i, j)] -= vr[i] * sr - vi[i] * si; hi[idx(i, j)] -= vr[i] * si + vi[i] * sr; }
    }
    for (let i = 0; i < n; i++) {                                     // H = H (I - 2 v v^H / v^H v)
      let sr = 0, si = 0;
      for (let j = k + 1; j < n; j++) { sr += hr[idx(i, j)] * vr[j] - hi[idx(i, j)] * vi[j]; si += hr[idx(i, j)] * vi[j] + hi[idx(i, j)] * vr[j]; }
      sr *= 2 / vn; si *= 2 / vn;
      for (let j = k + 1; j < n; j++) { hr[idx(i, j)] -= sr * vr[j] + si * vi[j]; hi[idx(i, j)] -= si * vr[j] - sr * vi[j]; }
    }
  }
  const ev = [];
  let top = n - 1, it = 0, total = 0, hn = 0;
  for (let i = 0; i < n * n; i++) hn += hr[i] * hr[i] + hi[i] * hi[i];
  hn = Math.sqrt(hn);                                                 // deflation is relative to |H| too:
                                                                      // near-zero diagonals (infinite pencil
                                                                      // eigenvalues) would otherwise never deflate
  const cs = new Float64Array(3 * n);
  while (top >= 0) {
    if (top === 0) { ev.push([hr[0], hi[0]]); break; }
    let l = top;
    for (; l > 0; l--) {
      const s = Math.hypot(hr[idx(l - 1, l - 1)], hi[idx(l - 1, l - 1)]) + Math.hypot(hr[idx(l, l)], hi[idx(l, l)]);
      if (Math.hypot(hr[idx(l, l - 1)], hi[idx(l, l - 1)]) <= Number.EPSILON * Math.max(s, 1e-3 * hn)) { hr[idx(l, l - 1)] = hi[idx(l, l - 1)] = 0; break; }
    }
    if (l === top) { ev.push([hr[idx(top, top)], hi[idx(top, top)]]); top--; it = 0; continue; }
    if (++total > 60 * n) {                                           // no convergence: return what remains
      for (let i = top; i >= 0; i--) ev.push([hr[idx(i, i)], hi[idx(i, i)]]);
      break;
    }
    it++;
    let mr, mi;
    if (it % 11 === 0) {                                              // exceptional shift
      mr = hr[idx(top, top)] + Math.abs(hr[idx(top, top - 1)]) * 1.5; mi = hi[idx(top, top)];
    } else {                                                          // Wilkinson: eigenvalue of the trailing 2x2 closest to d
      const ar = hr[idx(top - 1, top - 1)], ai = hi[idx(top - 1, top - 1)], br = hr[idx(top - 1, top)], bi = hi[idx(top - 1, top)];
      const cr = hr[idx(top, top - 1)], ci = hi[idx(top, top - 1)], dr = hr[idx(top, top)], di = hi[idx(top, top)];
      const hr2 = (ar - dr) / 2, hi2 = (ai - di) / 2;
      const qr = hr2 * hr2 - hi2 * hi2 + br * cr - bi * ci, qi = 2 * hr2 * hi2 + br * ci + bi * cr;
      const [sr, si] = csqrt(qr, qi);
      const mr1 = (ar + dr) / 2 + sr, mi1 = (ai + di) / 2 + si, mr2 = (ar + dr) / 2 - sr, mi2 = (ai + di) / 2 - si;
      if (Math.hypot(mr1 - dr, mi1 - di) <= Math.hypot(mr2 - dr, mi2 - di)) { mr = mr1; mi = mi1; } else { mr = mr2; mi = mi2; }
    }
    for (let i = l; i <= top; i++) { hr[idx(i, i)] -= mr; hi[idx(i, i)] -= mi; }
    for (let k = l; k < top; k++) {                                   // H - mu I = Q R by Givens, rows
      const ar = hr[idx(k, k)], ai = hi[idx(k, k)], br = hr[idx(k + 1, k)], bi = hi[idx(k + 1, k)];
      const na = Math.hypot(ar, ai), nb = Math.hypot(br, bi), nn = Math.hypot(na, nb);
      let c, sr, si;
      if (nn === 0) { c = 1; sr = si = 0; }
      else if (na === 0) { c = 0; sr = br / nb; si = -bi / nb; }
      else { c = na / nn; const pr = ar / na, pi = ai / na; sr = (pr * br + pi * bi) / nn; si = (pi * br - pr * bi) / nn; }
      cs[3 * k] = c; cs[3 * k + 1] = sr; cs[3 * k + 2] = si;
      for (let j = k; j <= top; j++) {
        const xr = hr[idx(k, j)], xi = hi[idx(k, j)], yr = hr[idx(k + 1, j)], yi = hi[idx(k + 1, j)];
        hr[idx(k, j)] = c * xr + sr * yr - si * yi; hi[idx(k, j)] = c * xi + sr * yi + si * yr;
        hr[idx(k + 1, j)] = -(sr * xr + si * xi) + c * yr; hi[idx(k + 1, j)] = -(sr * xi - si * xr) + c * yi;
      }
    }
    for (let k = l; k < top; k++) {                                   // R Q, columns
      const c = cs[3 * k], sr = cs[3 * k + 1], si = cs[3 * k + 2];
      for (let i = l; i <= Math.min(k + 2, top); i++) {
        const xr = hr[idx(i, k)], xi = hi[idx(i, k)], yr = hr[idx(i, k + 1)], yi = hi[idx(i, k + 1)];
        hr[idx(i, k)] = xr * c + yr * sr + yi * si; hi[idx(i, k)] = xi * c + yi * sr - yr * si;
        hr[idx(i, k + 1)] = -(xr * sr - xi * si) + c * yr; hi[idx(i, k + 1)] = -(xr * si + xi * sr) + c * yi;
      }
    }
    for (let i = l; i <= top; i++) { hr[idx(i, i)] += mr; hi[idx(i, i)] += mi; }
  }
  return ev;
}
function csqrt(re, im) {
  const r = Math.hypot(re, im);
  if (r === 0) return [0, 0];
  const sr = Math.sqrt((r + Math.abs(re)) / 2);
  return re >= 0 ? [sr, im / (2 * sr)] : [Math.abs(im) / (2 * sr), im >= 0 ? sr : -sr];
}

export function eigvalsPencil(A, B) {
  // finite eigenvalues l of A x = l B x by shift and invert: l = s + 1/m for the eigenvalues m of
  // (A - s B)^-1 B; m = 0 marks an infinite eigenvalue (singular B), which is dropped
  let s, M;
  for (s of SHIFTS) { M = caxpy(A, B, -s[0], -s[1]); if (ccond(M) < 1e10) break; }
  const m = ceigvals(csolve(M, B));
  const big = Math.max(...m.map(([r, i]) => Math.hypot(r, i)), 1e-300);
  return m.filter(([r, i]) => Math.hypot(r, i) > 1e-12 * big).map(([r, i]) => {
    const d = r * r + i * i;
    return [s[0] + r / d, s[1] - i / d];
  });
}

// ---------------------------------------------------------------- Raghavan-Roth quantities and DFT
function q14(M) {
  const l = [M[2], M[6], M[10]], p = [M[3], M[7], M[11]];
  const pp = p[0] * p[0] + p[1] * p[1] + p[2] * p[2], pl = p[0] * l[0] + p[1] * l[1] + p[2] * l[2];
  const px = cross(p, l);
  return [l[0], l[1], l[2], p[0], p[1], p[2], pp, pl, px[0], px[1], px[2],
    pp * l[0] - 2 * pl * p[0], pp * l[1] - 2 * pl * p[1], pp * l[2] - 2 * pl * p[2]];
}
function dft9(Y) {
  // Y[a][b]: 14 values on the 3x3 grid -> c[ka][kb] = {re, im} (14 each), with the 1/9 factor
  const c = [0, 1, 2].map(() => [0, 1, 2].map(() => ({ re: new Float64Array(14), im: new Float64Array(14) })));
  for (let ka = 0; ka < 3; ka++) for (let kb = 0; kb < 3; kb++) {
    const o = c[ka][kb];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      const [w1r, w1i] = W3[ka][a], [w2r, w2i] = W3[kb][b], wr = (w1r * w2r - w1i * w2i) / 9, wi = (w1r * w2i + w1i * w2r) / 9;
      const y = Y[a][b];
      for (let k = 0; k < 14; k++) { o.re[k] += y[k] * wr; o.im[k] += y[k] * wi; }
    }
  }
  return c;
}
function coeffsFromGrid(Y) {
  // 14 x 9 coefficients for the monomials s_a s_b, s_a c_b, c_a s_b, c_a c_b, s_a, c_a, s_b, c_b, 1
  const c = dft9(Y), pp = c[1][1], pm = c[1][2], p0 = c[1][0], q0 = c[0][1], c00 = c[0][0];
  const C = new Float64Array(14 * 9);
  for (let k = 0; k < 14; k++) {
    const row = [-2 * (pp.re[k] - pm.re[k]), -2 * (pp.im[k] + pm.im[k]), -2 * (pp.im[k] - pm.im[k]), 2 * (pp.re[k] + pm.re[k]),
      -2 * p0.im[k], 2 * p0.re[k], -2 * q0.im[k], 2 * q0.re[k], c00.re[k]];
    C.set(row, 9 * k);
  }
  return C;
}
function leftNull(Q, m, n) {
  // orthonormal basis of the left null space of the m x n real matrix Q (rank n), by Householder QR
  const A = Float64Array.from(Q), Qm = new Float64Array(m * m);
  for (let i = 0; i < m; i++) Qm[i * m + i] = 1;
  for (let k = 0; k < n; k++) {
    let nx = 0;
    for (let i = k; i < m; i++) nx += A[i * n + k] ** 2;
    nx = Math.sqrt(nx);
    const v = new Float64Array(m);
    for (let i = k; i < m; i++) v[i] = A[i * n + k];
    v[k] += (v[k] >= 0 ? 1 : -1) * nx;
    let vn = 0;
    for (let i = k; i < m; i++) vn += v[i] * v[i];
    if (vn === 0) continue;
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let i = k; i < m; i++) s += v[i] * A[i * n + j];
      s *= 2 / vn;
      for (let i = k; i < m; i++) A[i * n + j] -= s * v[i];
    }
    for (let i = 0; i < m; i++) {
      let s = 0;
      for (let j = k; j < m; j++) s += Qm[i * m + j] * v[j];
      s *= 2 / vn;
      for (let j = k; j < m; j++) Qm[i * m + j] -= s * v[j];
    }
  }
  const N = new Float64Array((m - n) * m);                          // rows: the last m - n columns of Q
  for (let r = 0; r < m - n; r++) for (let k = 0; k < m; k++) N[r * m + k] = Qm[k * m + n + r];
  return N;
}

// ---------------------------------------------------------------- one elimination route
const RZ_NEG = G3.map(a => rz(-a));
const P3 = [2, 0, 1];                                               // DFT index order -1, 0, 1

class Route {
  constructor(Fs, B, T) {
    this.Fs = Fs; this.B = B; this.T = T;
    const Ti = mul4s(inv4(B), T, inv4(Fs[5]));
    const F1i = inv4(Fs[0]), F2i = inv4(Fs[1]);
    const M1 = RZ_NEG.map(R => mul4s(F1i, R, Ti)), M2 = RZ_NEG.map(R => mul4(F2i, R));
    const Q = coeffsFromGrid(M1.map(m1 => M2.map(m2 => q14(mul4(m2, m1)))));
    this.Q8 = new Float64Array(14 * 8); this.qc = new Float64Array(14);
    for (let k = 0; k < 14; k++) { for (let j = 0; j < 8; j++) this.Q8[8 * k + j] = Q[9 * k + j]; this.qc[k] = Q[9 * k + 8]; }
    this.N = leftNull(this.Q8, 14, 8);                               // 6 x 14, N Q8 = 0
    this.G45 = G3.map(a => G3.map(b => mul4s(rz(a), Fs[3], rz(b), Fs[4])));
    this.F6i = inv4(Fs[5]);
    this._S = null; this._Q8pinv = null;
  }

  sigma(t3) {                                                        // 12x12 Sigma(t3), basis z4^i z5^j
    const R = mul4(rz(t3), this.Fs[2]);
    const c = dft9(this.G45.map(row => row.map(G => q14(mul4(R, G)))));
    for (let k = 0; k < 14; k++) c[0][0].re[k] -= this.qc[k];
    const S = cmat(12, 12);
    for (let r = 0; r < 6; r++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
      const o = c[P3[a]][P3[b]];
      let sr = 0, si = 0;
      for (let k = 0; k < 14; k++) { sr += this.N[14 * r + k] * o.re[k]; si += this.N[14 * r + k] * o.im[k]; }
      S.re[12 * r + 3 * a + b] = sr; S.im[12 * r + 3 * a + b] = si;               // rows 0-5
      S.re[12 * (r + 6) + 3 * (a + 1) + b] = sr; S.im[12 * (r + 6) + 3 * (a + 1) + b] = si;   // the copy times z4
    }
    return S;
  }

  rawS() {                                                           // S-, S0, S+ by a 3-point DFT
    if (!this._S) {
      const sig = G3.map(x => this.sigma(x));
      const c = [0, 1, 2].map(k => {
        let C = cmat(12, 12);
        for (let x = 0; x < 3; x++) C = caxpy(C, sig[x], W3[k][x][0] / 3, W3[k][x][1] / 3);
        return C;
      });
      this._S = [c[2], c[0], c[1]];
    }
    return this._S;
  }

  usable(relTol = 1e-9) {                                            // rank Q = 8 and rank Sigma = 12
    const q = csvd(cfromReal(14, 8, this.Q8)).s, qmax = maxAbs(this.Q8);
    if (q.filter(v => v > relTol * qmax).length < 8) return false;
    const [Sm, S0, Sp] = this.rawS();
    for (const t of SELECT_T) {
      const M = caxpy(caxpy(S0, Sm, Math.cos(t), -Math.sin(t)), Sp, Math.cos(t), Math.sin(t));
      const s = csvd(M).s;
      if (s.filter(v => v / s[0] > relTol).length < 12) return false;
    }
    return true;
  }

  rootsQ3(rootTol = 2e-3, same = 1e-9) {
    const [Sm, S0, Sp] = this.rawS();
    const sc = Math.max(cabsMax(Sm), cabsMax(S0), cabsMax(Sp));
    const A = cscale(Sp, 1 / sc), Bm = cscale(S0, 1 / sc), Cm = cscale(Sm, 1 / sc), I = ceye(12), Z = cmat(12, 12);
    let z;
    if (ccond(A) < 1e8) z = ceigvals(cblock([[Z, I], [cscale(csolve(A, Cm), -1), cscale(csolve(A, Bm), -1)]]));
    else z = eigvalsPencil(cblock([[Z, I], [cscale(Cm, -1), cscale(Bm, -1)]]), cblock([[I, Z], [Z, A]]));
    const th = z.filter(([r, i]) => Number.isFinite(r) && Number.isFinite(i) && Math.abs(Math.hypot(r, i) - 1) < rootTol)
      .map(([r, i]) => Math.atan2(i, r)).sort((x, y) => x - y);
    return th.filter((t, k) => k === 0 || t - th[k - 1] > same);
  }

  rootsQ4Q5(t3s, rootTol = 2e-3, singTol = 1e-6) {
    const [Sm, S0, Sp] = this.rawS(), out = [], I = ceye(3), Z = cmat(3, 3);
    const R3m = cfromReal(3, 6, R3.flat());
    for (const t3 of t3s) {
      const er = Math.cos(t3), ei = Math.sin(t3);
      const C = [0, 1, 2].map(() => cmat(6, 3));                    // C[a] (6x3): coefficient of z4^a
      for (let r = 0; r < 6; r++) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
        const k = 12 * r + 3 * a + b;
        C[a].re[3 * r + b] = Sm.re[k] * er + Sm.im[k] * ei + S0.re[k] + Sp.re[k] * er - Sp.im[k] * ei;
        C[a].im[3 * r + b] = Sm.im[k] * er - Sm.re[k] * ei + S0.im[k] + Sp.re[k] * ei + Sp.im[k] * er;
      }
      const [A0, A1, A2] = C.map(X => cmul(R3m, X));
      let z4;
      if (ccond(A2) < 1e8) z4 = ceigvals(cblock([[Z, I], [cscale(csolve(A2, A0), -1), cscale(csolve(A2, A1), -1)]]));
      else z4 = eigvalsPencil(cblock([[Z, I], [cscale(A0, -1), cscale(A1, -1)]]), cblock([[I, Z], [Z, A2]]));
      for (const [zr0, zi0] of z4) {
        if (!(Number.isFinite(zr0) && Number.isFinite(zi0)) || Math.abs(Math.hypot(zr0, zi0) - 1) >= rootTol) continue;
        const zn = Math.hypot(zr0, zi0), zr = zr0 / zn, zi = zi0 / zn;
        const Cz = caxpy(caxpy(C[0], C[1], zr, zi), C[2], zr * zr - zi * zi, 2 * zr * zi);
        const { s, U } = csvd(Cz);
        if (s[2] > singTol * s[0]) continue;                          // a root of the projection only
        const c = [0, 1, 2].map(j => {                                // U[:, 0]^H Cz: coefficients of 1, z5, z5^2
          let sr = 0, si = 0;
          for (let i = 0; i < 6; i++) {
            const ur = U.re[3 * i], ui = -U.im[3 * i], xr = Cz.re[3 * i + j], xi = Cz.im[3 * i + j];
            sr += ur * xr - ui * xi; si += ur * xi + ui * xr;
          }
          return [sr, si];
        });
        const disc = csqrt(c[1][0] ** 2 - c[1][1] ** 2 - 4 * (c[2][0] * c[0][0] - c[2][1] * c[0][1]),
          2 * c[1][0] * c[1][1] - 4 * (c[2][0] * c[0][1] + c[2][1] * c[0][0]));
        const den = [2 * c[2][0], 2 * c[2][1]], dd = den[0] ** 2 + den[1] ** 2;
        for (const sg of [1, -1]) {
          const nr = -c[1][0] + sg * disc[0], ni = -c[1][1] + sg * disc[1];
          const wr0 = (nr * den[0] + ni * den[1]) / dd, wi0 = (ni * den[0] - nr * den[1]) / dd;
          if (!(Number.isFinite(wr0) && Number.isFinite(wi0)) || Math.abs(Math.hypot(wr0, wi0) - 1) >= rootTol) continue;
          const wn = Math.hypot(wr0, wi0), wr = wr0 / wn, wi = wi0 / wn;
          const y = [[1, 0], [wr, wi], [wr * wr - wi * wi, 2 * wr * wi]];
          let res = 0;
          for (let i = 0; i < 6; i++) {
            let sr = 0, si = 0;
            for (let j = 0; j < 3; j++) {
              const xr = Cz.re[3 * i + j], xi = Cz.im[3 * i + j];
              sr += xr * y[j][0] - xi * y[j][1]; si += xr * y[j][1] + xi * y[j][0];
            }
            res += sr * sr + si * si;
          }
          if (Math.sqrt(res) <= 1e-6 * s[0] * Math.sqrt(3)) out.push([t3, Math.atan2(zi, zr), Math.atan2(wi, wr)]);
        }
      }
    }
    return out;
  }

  complete(roots) {                                                  // (q1, q2), q6, Newton polish
    const F = this.Fs;
    if (!this._Q8pinv) this._Q8pinv = cpinv(cfromReal(14, 8, this.Q8)).re;
    const P = this._Q8pinv, TF6 = mul4(this.T, this.F6i);
    return roots.map(([t3, t4, t5]) => {
      const rhs = q14(mul4s(rz(t3), F[2], rz(t4), F[3], rz(t5), F[4])).map((v, k) => v - this.qc[k]);
      const m = [0, 1, 2, 3, 4, 5, 6, 7].map(i => rhs.reduce((s, v, k) => s + P[14 * i + k] * v, 0));
      const q = [Math.atan2(m[4], m[5]), Math.atan2(m[6], m[7]), t3, t4, t5, 0];
      const A15 = fk(q.slice(0, 5), F, this.B);
      let r00 = 0, r10 = 0;                                          // R6 = A15_R^T (T F6^-1)_R, entries (0,0), (1,0)
      for (let k = 0; k < 3; k++) { r00 += A15[4 * k] * TF6[4 * k]; r10 += A15[4 * k + 1] * TF6[4 * k]; }
      q[5] = Math.atan2(r10, r00);
      return polish(q, F, this.B, this.T);
    });
  }

  solve() {
    const r = this.rootsQ4Q5(this.rootsQ3());
    return r.length ? this.complete(r) : [];
  }
}

function polish(q, Fs, B, T, iters = 6, tol = 2e-15) {
  // a few Newton steps on the 12 pose residuals with the analytic Jacobian
  q = q.slice();
  for (let it = 0; it < iters; it++) {
    let M = B;
    const W = [], O = [];
    for (let j = 0; j < 6; j++) {
      W.push([M[2], M[6], M[10]]); O.push([M[3], M[7], M[11]]);
      M = mul4(mul4(M, rz(q[j])), Fs[j]);
    }
    const r = new Float64Array(12), p = [M[3], M[7], M[11]];
    for (let i = 0; i < 3; i++) { r[i] = p[i] - T[4 * i + 3]; for (let j = 0; j < 3; j++) r[3 + 3 * i + j] = M[4 * i + j] - T[4 * i + j]; }
    if (maxAbs(r) < tol) break;
    const J = new Float64Array(72);
    for (let k = 0; k < 6; k++) {
      const w = W[k], d = [p[0] - O[k][0], p[1] - O[k][1], p[2] - O[k][2]], v = cross(w, d);
      J[k] = v[0]; J[6 + k] = v[1]; J[12 + k] = v[2];
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {      // ([w]x R)_{ij}
        const wr = [w[1] * M[8 + j] - w[2] * M[4 + j], w[2] * M[j] - w[0] * M[8 + j], w[0] * M[4 + j] - w[1] * M[j]];
        J[6 * (3 + 3 * i + j) + k] = wr[i];
      }
    }
    const P = cpinv(cfromReal(12, 6, J)).re;
    let big = 0;
    for (let k = 0; k < 6; k++) {
      let dq = 0;
      for (let i = 0; i < 12; i++) dq += P[12 * k + i] * r[i];
      q[k] -= dq; big = Math.max(big, Math.abs(dq));
    }
    if (big < 1e-15) break;
  }
  return q;
}

function routes(Fs, B, T) {
  // the 12 equivalent write-ups of the loop, each [Fs', B', T', back], back(q') -> q
  const I4 = eye4(), out = [];
  for (const rev of [false, true]) {
    let G, Bg, Tg, back;
    if (!rev) { G = Fs.slice(); Bg = B; Tg = T; back = q => q; }
    else {
      const Fi = Fs.map(inv4);
      G = [0, 1, 2, 3, 4].map(i => Fi[4 - i]).concat([inv4(B)]);
      Bg = Fi[5]; Tg = inv4(T); back = q => q.slice().reverse().map(v => -v);
    }
    for (let k = 0; k < 6; k++) {
      if (k === 0) { out.push([G, Bg, Tg, back]); continue; }
      const Gk = G.slice(k, 5).concat([mul4s(G[5], inv4(Tg), Bg)], G.slice(0, k));
      const perm = [...Array(6).keys()].map(i => (i + k) % 6), bk = back;
      out.push([Gk, I4, I4, q => { const qo = new Array(6); perm.forEach((p, i) => { qo[p] = q[i]; }); return bk(qo); }]);
    }
  }
  return [0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11].map(i => out[i]);
}

function chainToBF(elems, locked) {
  // elems: {type: 'fixed', tf} or {type: 'rev', axis, name}, walked at zero joint angles
  let tf = eye4(), k = 0;
  const G = [], names = [];
  for (const e of elems) {
    if (e.type === 'fixed') { tf = mul4(tf, toM4(e.tf)); continue; }
    let c = null;
    if (locked) c = locked[k] ?? locked[e.name] ?? null;
    k++;
    if (c !== null) tf = mul4(tf, rotation(e.axis, c));
    else {
      const R = [0, 1, 2].map(i => tf[4 * i] * e.axis[0] + tf[4 * i + 1] * e.axis[1] + tf[4 * i + 2] * e.axis[2]);
      G.push(frameOnAxis([tf[3], tf[7], tf[11]], R)); names.push(e.name);
    }
  }
  if (G.length !== 6) throw new Error(`TrigoIK needs exactly 6 free revolute joints, got ${G.length} (hold the extra joints with locked)`);
  const Fs = [0, 1, 2, 3, 4].map(i => mul4(inv4(G[i]), G[i + 1])).concat([mul4(inv4(G[5]), tf)]);
  return { B: G[0], Fs, names };
}

function parseURDF(text, baseLink, tipLink) {
  // joints from baseLink to tipLink as fixed / rev elements; a small parser for the URDF subset used
  const attr = (s, name) => { const m = s.match(new RegExp(`\\b${name}\\s*=\\s*["']([^"']*)["']`)); return m ? m[1] : null; };
  const tag = (body, name) => { const m = body.match(new RegExp(`<${name}\\b([^>]*)/?>`)); return m ? m[1] : null; };
  const nums = (s, d) => (s ?? d).trim().split(/\s+/).map(Number);
  const joints = {};
  for (const m of text.replace(/<!--[\s\S]*?-->/g, '').matchAll(/<joint\b([^>]*)>([\s\S]*?)<\/joint>/g)) {
    const body = m[2], o = tag(body, 'origin'), a = tag(body, 'axis');
    const j = { name: attr(m[1], 'name'), type: attr(m[1], 'type'), parent: attr(tag(body, 'parent'), 'link'),
      child: attr(tag(body, 'child'), 'link'), xyz: nums(o && attr(o, 'xyz'), '0 0 0'), rpy: nums(o && attr(o, 'rpy'), '0 0 0'),
      axis: nums(a && attr(a, 'xyz'), '1 0 0') };
    joints[j.child] = j;
  }
  const parents = new Set(Object.values(joints).map(j => j.parent));
  if (!baseLink) {
    const roots = [...parents].filter(l => !(l in joints));
    if (roots.length !== 1) throw new Error(`URDF: give baseLink, candidates ${roots}`);
    baseLink = roots[0];
  }
  if (!tipLink) {
    const leaves = Object.keys(joints).filter(l => !parents.has(l));
    if (leaves.length !== 1) throw new Error(`URDF: give tipLink, candidates ${leaves}`);
    tipLink = leaves[0];
  }
  const path = [];
  for (let link = tipLink; link !== baseLink; link = joints[link].parent) {
    if (!(link in joints)) throw new Error(`URDF: no path from ${baseLink} to ${tipLink}`);
    path.push(joints[link]);
  }
  const elems = [];
  for (const j of path.reverse()) {
    elems.push({ type: 'fixed', tf: mul4(translation(...j.xyz), rpy(...j.rpy)) });
    if (j.type === 'revolute' || j.type === 'continuous') elems.push({ type: 'rev', axis: j.axis, name: j.name });
    else if (j.type !== 'fixed') throw new Error(`URDF: joint ${j.name} of type ${j.type} is not supported`);
  }
  return elems;
}

// ---------------------------------------------------------------- the solver
export class TrigoIK {
  // All IK solutions of a 6R chain  FK(q) = B Rz(q1) F1 Rz(q2) F2 ... Rz(q6) F6.
  // Build it with TrigoIK.fromDH, TrigoIK.fromURDF, TrigoIK.fromElements, or from B and F1..F6
  // (4x4 as nested arrays or 16 numbers, row-major).  The elimination route is selected once, here.
  constructor(B, Fs, { tol = 1e-8, jointNames = null } = {}) {
    this.B = toM4(B);
    this.Fs = Fs.map(toM4);
    if (this.Fs.length !== 6) throw new Error('TrigoIK needs F1..F6');
    this.tol = tol;
    this.jointNames = jointNames;
    this.route = this._selectRoute();                                 // null: continuous self-motion
  }

  static fromDH({ d, a, alpha, theta0 = null, modified = false, locked = null, base = null, tool = null }, opts = {}) {
    // standard: A_i = Rz(q_i + theta0_i) Tz(d_i) Tx(a_i) Rx(alpha_i)
    // modified (Craig): A_i = Rx(alpha_i) Tx(a_i) Rz(q_i + theta0_i) Tz(d_i)
    theta0 = theta0 ?? d.map(() => 0);
    const elems = [{ type: 'fixed', tf: base ? toM4(base) : eye4() }];
    for (let i = 0; i < d.length; i++) {
      if (modified) elems.push({ type: 'fixed', tf: mul4(rotation([1, 0, 0], alpha[i]), translation(a[i])) },
        { type: 'rev', axis: [0, 0, 1], name: i }, { type: 'fixed', tf: mul4(rz(theta0[i]), translation(0, 0, d[i])) });
      else elems.push({ type: 'rev', axis: [0, 0, 1], name: i },
        { type: 'fixed', tf: mul4s(rz(theta0[i]), translation(0, 0, d[i]), translation(a[i]), rotation([1, 0, 0], alpha[i])) });
    }
    if (tool) elems.push({ type: 'fixed', tf: toM4(tool) });
    return TrigoIK.fromElements(elems, locked, opts);
  }

  static fromURDF(text, { baseLink = null, tipLink = null, locked = null } = {}, opts = {}) {
    // text of a URDF file; the chain from baseLink to tipLink (default: the root and the single leaf)
    return TrigoIK.fromElements(parseURDF(text, baseLink, tipLink), locked, opts);
  }

  static fromElements(elems, locked = null, opts = {}) {
    // a serial chain as {type: 'fixed', tf: 4x4} / {type: 'rev', axis: [x, y, z], name} elements;
    // locked = {joint name or index among the revolute joints: angle}
    const { B, Fs, names } = chainToBF(elems, locked);
    return new TrigoIK(B, Fs, { jointNames: names, ...opts });
  }

  get selfMotion() { return this.route === null; }

  fk(q) { return toNested(fk(q, this.Fs, this.B)); }

  ik(T) {
    // all solutions for the tip pose T (4x4), as an array of 6-vectors with angles in [-pi, pi)
    if (this.route === null) throw new SelfMotionError();
    T = toM4(T);
    const R = cmat(3, 3);                                            // re-orthonormalize the target
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) R.re[3 * i + j] = T[4 * i + j];
    const { U, V } = csvd(R);
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += U.re[3 * i + k] * V.re[3 * j + k];
      T[4 * i + j] = s;
    }
    const [G, Bg, Tg, back] = routes(this.Fs, this.B, T)[this.route];
    const out = [];
    for (const q0 of new Route(G, Bg, Tg).solve()) {
      const q = back(q0);
      if (maxAbs(poseRes(q, this.Fs, this.B, T)) >= this.tol) continue;
      const w = q.map(wrap);
      if (out.every(u => Math.hypot(...u.map((v, k) => wrap(w[k] - v))) > 1e-4)) out.push(w);
    }
    return out;
  }

  residual(q, T) {
    // max abs pose residual of one solution q (6) or of each of several (n x 6)
    T = toM4(T);
    return Array.isArray(q[0]) ? q.map(x => maxAbs(poseRes(x, this.Fs, this.B, T))) : maxAbs(poseRes(q, this.Fs, this.B, T));
  }

  _selectRoute() {
    // once per robot: the first of the twelve routes with rank Q = 8 and rank Sigma = 12 (the target
    // of a fixed random configuration; degeneracy depends on the geometry only)
    const T = fk(SELECT_Q, this.Fs, this.B);
    const rs = routes(this.Fs, this.B, T);
    for (let j = 0; j < rs.length; j++) if (new Route(rs[j][0], rs[j][1], rs[j][2]).usable()) return j;
    return null;
  }
}
