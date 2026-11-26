// All IK solutions of a target, one view per solution, all views share one camera.
// One WebGL canvas covers the grid; each cell is drawn with a scissor viewport.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = id => document.getElementById(id);
const stage = $('stage'), grid = $('grid'), canvas = $('gl'), info = $('info'), loading = $('loading');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setScissorTest(true);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(2, -1.5, 3);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xffffff, 0.6);
fill.position.set(-2, 2, 1);
scene.add(fill);

const camera = new THREE.PerspectiveCamera(30, 1, 0.01, 50);
camera.up.set(0, 0, 1);
const controls = new OrbitControls(camera, grid);
controls.enableDamping = true;
controls.dampingFactor = 0.12;

const CELL_BG = new THREE.Color(0xf3f5f7);
let robots = [], cur = null;          // cur: {meta, root, linkObjs, group, target}
const cache = {};

// ------------------------------------------------------------------ data
async function gunzip(buf) {
  const u8 = new Uint8Array(buf);
  if (!(u8[0] === 0x1f && u8[1] === 0x8b)) return buf;            // already decoded by the server
  const ds = new DecompressionStream('gzip');
  return await new Response(new Blob([buf]).stream().pipeThrough(ds)).arrayBuffer();
}

async function loadRobot(key) {
  if (cache[key]) return cache[key];
  const [meta, raw] = await Promise.all([
    fetch(`data/${key}.json`).then(r => r.json()),
    fetch(`data/${key}.bin.gz`).then(r => r.arrayBuffer()).then(gunzip),
  ]);
  const root = new THREE.Group();
  const linkObjs = meta.links.map(() => {
    const g = new THREE.Group();
    g.matrixAutoUpdate = false;
    root.add(g);
    return g;
  });
  for (const p of meta.parts) {
    const q = new Uint16Array(raw, p.pos, p.nv * 3);
    const pos = new Float32Array(p.nv * 3);
    for (let i = 0; i < p.nv; i++)
      for (let k = 0; k < 3; k++) pos[3 * i + k] = p.lo[k] + q[3 * i + k] / 65535 * p.scale[k];
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const nrm = new Int8Array(raw, p.nrm, p.nv * 3);
    if (nrm.some(v => v !== 0)) geo.setAttribute('normal', new THREE.BufferAttribute(nrm, 3, true));
    geo.setIndex(new THREE.BufferAttribute(
      p.idx32 ? new Uint32Array(raw, p.idx, p.ni) : new Uint16Array(raw, p.idx, p.ni), 1));
    if (!geo.getAttribute('normal')) geo.computeVertexNormals();
    const col = new THREE.Color().setRGB(p.rgb[0], p.rgb[1], p.rgb[2], THREE.SRGBColorSpace);
    const mat = new THREE.MeshStandardMaterial({
      color: col, roughness: 0.55, metalness: 0.1,
      transparent: p.alpha < 1, opacity: p.alpha, side: THREE.DoubleSide,
    });
    linkObjs[p.link].add(new THREE.Mesh(geo, mat));
  }
  meta.tf0 = meta.links.map(L => new THREE.Matrix4().set(...L.tf0.flat()));
  meta.axis = meta.links.map(L => new THREE.Vector3(...L.axis).normalize());
  return (cache[key] = { meta, root, linkObjs });
}

// ------------------------------------------------------------------ kinematics
const _R = new THREE.Matrix4();
function setPose(r, q) {
  const { meta, linkObjs } = r;
  meta.links.forEach((L, i) => {
    const m = linkObjs[i].matrix;
    if (L.parent < 0) { m.identity(); }
    else {
      m.copy(linkObjs[L.parent].matrix).multiply(meta.tf0[i]);
      m.multiply(_R.makeRotationAxis(meta.axis[i], q[L.q]));
    }
    linkObjs[i].matrixWorldNeedsUpdate = true;
  });
}

function fitCamera(r, sols) {
  // corners of every mesh box in every solution, then the distance at which all of them fit
  const pts = [], box = new THREE.Box3(), b = new THREE.Box3();
  for (const q of sols) {
    setPose(r, q);
    r.root.updateMatrixWorld(true);
    r.root.traverse(o => {
      if (!o.isMesh) return;
      if (!o.geometry.boundingBox) o.geometry.computeBoundingBox();
      b.copy(o.geometry.boundingBox);
      for (let k = 0; k < 8; k++)
        pts.push(new THREE.Vector3(k & 1 ? b.max.x : b.min.x, k & 2 ? b.max.y : b.min.y,
          k & 4 ? b.max.z : b.min.z).applyMatrix4(o.matrixWorld));
    });
  }
  box.setFromPoints(pts);
  const c = box.getCenter(new THREE.Vector3());
  const back = new THREE.Vector3(1.0, -1.25, 0.6).normalize();            // from target to camera
  const right = new THREE.Vector3().crossVectors(camera.up, back).normalize();
  const up = new THREE.Vector3().crossVectors(back, right);
  const t = Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * 0.84;   // margin
  let d = 0;
  for (const p of pts) {
    const v = p.clone().sub(c);
    const f = v.dot(back);                                               // towards the camera
    d = Math.max(d, f + Math.abs(v.dot(up)) / t, f + Math.abs(v.dot(right)) / t);
  }
  controls.target.copy(c);
  camera.position.copy(c).add(back.multiplyScalar(d));
  camera.near = d / 50; camera.far = d * 10;
  camera.updateProjectionMatrix();
  controls.update();
}

// ------------------------------------------------------------------ UI
function chips(el, items, onPick, active) {
  el.innerHTML = '';
  items.forEach((it, i) => {
    const b = document.createElement('button');
    b.className = 'chip' + (i === active ? ' on' : '');
    b.textContent = it.label;
    b.disabled = !!it.disabled;
    if (it.title) b.title = it.title;
    b.onclick = () => {
      [...el.children].forEach(c => c.classList.remove('on'));
      b.classList.add('on');
      onPick(i);
    };
    el.appendChild(b);
  });
}

const fmtRes = x => x.toExponential(0).replace('e-', '×10⁻').replace(/\d+$/, m =>
  [...m].map(d => '⁰¹²³⁴⁵⁶⁷⁸⁹'[d]).join(''));

function histText(h) {
  const tot = Object.values(h).reduce((a, b) => a + b, 0);
  return Object.entries(h).map(([k, v]) => `${k}: ${(100 * v / tot).toFixed(v / tot < 0.01 ? 1 : 0)}%`).join(', ');
}

function showTarget(t) {
  cur.target = t;
  grid.innerHTML = '';
  t.sols.forEach((_, i) => {
    const d = document.createElement('div');
    d.className = 'cell';
    d.innerHTML = `<span>Solution ${i + 1}</span>`;
    grid.appendChild(d);
  });
  const n = t.sols.length;
  grid.style.gridTemplateColumns = window.innerWidth <= 820 ? 'repeat(2, 1fr)'
    : `repeat(${n <= 2 ? 2 : n <= 9 && n % 3 === 0 ? 3 : 4}, 1fr)`;
  const total = Object.values(t.hist).reduce((a, b) => a + b, 0);
  info.innerHTML = `<b>${n} solutions</b>, largest pose residual ${fmtRes(t.residual)}. ` +
    `<span class="muted">Over ${total} random targets: ${histText(t.hist)}.</span>`;
  fitCamera(cur, t.sols);
  draw();
}

function showGroup(g) {
  if (g.self_motion) {
    grid.innerHTML = `<div class="selfmotion">With ${g.name.replace(' held', '')} held, all twelve elimination routes
      are degenerate: the remaining chain has continuous self-motion, and the solver reports it.</div>`;
    $('targets').innerHTML = '';
    info.innerHTML = '<b>Self-motion detected</b>';
    cur.target = null;
    draw();
    return;
  }
  const labels = { most: 'Most solutions', typical: 'Typical' };
  chips($('targets'), g.targets.map(t => ({ label: `${labels[t.tag]} (${t.n})` })), i => showTarget(g.targets[i]), 0);
  showTarget(g.targets[0]);
}

async function pickRobot(i) {
  loading.style.display = 'flex';
  const r = await loadRobot(robots[i].key);
  loading.style.display = 'none';
  if (cur) scene.remove(cur.root);
  cur = r;
  scene.add(r.root);
  const gs = r.meta.groups;
  $('heldRow').style.display = r.meta.axes === 7 ? '' : 'none';
  if (r.meta.axes === 7) {
    chips($('groups'), gs.map(g => ({
      label: g.name.replace(' held', '') + (g.self_motion ? ' (∞)' : ''),
      title: g.self_motion ? 'continuous self-motion' : `held at ${(g.held_angle * 180 / Math.PI).toFixed(1)}°`,
    })), k => showGroup(gs[k]), 0);
  }
  showGroup(gs[0]);
}

// ------------------------------------------------------------------ render
function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) ||
      canvas.height !== Math.floor(h * renderer.getPixelRatio()))
    renderer.setSize(w, h, false);
}

function draw() {
  resize();
  renderer.setScissor(0, 0, stage.clientWidth, stage.clientHeight);
  renderer.setClearColor(0xffffff, 0);
  renderer.clear();
  if (!cur || !cur.target) return;
  const H = stage.clientHeight, base = stage.getBoundingClientRect();
  const cells = grid.querySelectorAll('.cell');
  cells.forEach((cell, i) => {
    const r = cell.getBoundingClientRect();
    const x = r.left - base.left, y = H - (r.bottom - base.top);
    renderer.setViewport(x, y, r.width, r.height);
    renderer.setScissor(x, y, r.width, r.height);
    renderer.setClearColor(CELL_BG, 1);
    camera.aspect = r.width / r.height;
    camera.updateProjectionMatrix();
    setPose(cur, cur.target.sols[i]);
    cur.root.updateMatrixWorld(true);
    renderer.render(scene, camera);
  });
}

controls.addEventListener('change', draw);
window.addEventListener('resize', () => { if (cur && cur.target) showTarget(cur.target); else draw(); });
(function tick() { requestAnimationFrame(tick); controls.update(); })();

robots = await fetch('data/robots.json').then(r => r.json());
const d6 = robots.filter(r => r.axes === 6), d7 = robots.filter(r => r.axes === 7);
robots = [...d6, ...d7];
chips($('robots'), robots.map(r => ({ label: r.label })), pickRobot, 0);
pickRobot(0);
