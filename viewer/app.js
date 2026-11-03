// 3-D replay viewer for DDPG arm path planning (Three.js).
// Data: <run>/viewer/manifest.json + ckpt_XXXXXX.json written by arm_rl/export.py.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ------------------------------------------------------------------ helpers
const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const RUN = (params.get('run') || 'runs/ddpg_paper').replace(/\/+$/, '');
const DATA = `../${RUN}/viewer/`;
const toV = (p) => new THREE.Vector3(p[0], p[2], -p[1]); // z-up (sim) -> y-up (three)
const lerp = (a, b, t) => a + (b - a) * t;
const fmt = (x, d = 3) => (x === undefined || x === null || Number.isNaN(x) ? '—' : Number(x).toFixed(d));
const toast = (msg, ms = 2200) => { const t = $('toast'); t.textContent = msg; t.classList.remove('hidden'); clearTimeout(toast._h); toast._h = setTimeout(() => t.classList.add('hidden'), ms); };

// ------------------------------------------------------------------ renderer / scene
const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0b1222, 0.16);
const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 50);
camera.position.set(-0.55, 1.0, 1.85);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0.32, 0.32, 0.0);
controls.enableDamping = true; controls.dampingFactor = 0.06;
controls.minDistance = 0.4; controls.maxDistance = 8; controls.maxPolarAngle = Math.PI * 0.49;
controls.autoRotateSpeed = 0.6;

// lights
scene.add(new THREE.HemisphereLight(0x9fb7ff, 0x1a2238, 0.55));
const key = new THREE.DirectionalLight(0xffffff, 2.2);
key.position.set(1.6, 2.6, 1.2); key.castShadow = true;
key.shadow.mapSize.set(2048, 2048); key.shadow.bias = -0.0004; key.shadow.normalBias = 0.01;
Object.assign(key.shadow.camera, { left: -1.6, right: 1.6, top: 1.6, bottom: -1.6, near: 0.5, far: 8 });
scene.add(key);
const rim = new THREE.PointLight(0x60a5fa, 6, 4, 2); rim.position.set(-1.2, 1.0, -1.0); scene.add(rim);
const warm = new THREE.PointLight(0xf59e0b, 2.5, 3, 2); warm.position.set(0.8, 0.5, 1.4); scene.add(warm);

// ground
const groundMat = new THREE.MeshStandardMaterial({ color: 0x111a2c, roughness: 0.95, metalness: 0.05 });
const ground = new THREE.Mesh(new THREE.CircleGeometry(2.2, 96), groundMat);
ground.rotation.x = -Math.PI / 2; ground.receiveShadow = true; scene.add(ground);
const grid = new THREE.GridHelper(4, 40, 0x2a3a5c, 0x1b2740);
grid.material.transparent = true; grid.material.opacity = 0.55; grid.position.y = 0.001; scene.add(grid);
const ring = new THREE.Mesh(new THREE.RingGeometry(0.60, 0.615, 128), new THREE.MeshBasicMaterial({ color: 0x60a5fa, transparent: true, opacity: 0.25, side: THREE.DoubleSide }));
ring.rotation.x = -Math.PI / 2; ring.position.y = 0.002; scene.add(ring); // nominal reach footprint

// post-processing (soft bloom on emissives)
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.35, 0.6, 0.88);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h, false); composer.setSize(w, h);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

// ------------------------------------------------------------------ materials
const matLink = new THREE.MeshPhysicalMaterial({ color: 0x3b82f6, metalness: 0.55, roughness: 0.32, clearcoat: 0.7, clearcoatRoughness: 0.25 });
const matLinkHit = new THREE.MeshPhysicalMaterial({ color: 0xef4444, emissive: 0x7f1d1d, emissiveIntensity: 0.8, metalness: 0.4, roughness: 0.4 });
const matJoint = new THREE.MeshStandardMaterial({ color: 0x1e3a8a, metalness: 0.7, roughness: 0.3 });
const matBase = new THREE.MeshStandardMaterial({ color: 0x334155, metalness: 0.6, roughness: 0.45 });
const matEE = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, metalness: 0.9, roughness: 0.2 });
const matObs = new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.6, metalness: 0.1 });
const matTrunk = new THREE.MeshStandardMaterial({ color: 0x16a34a, roughness: 0.75, metalness: 0.05 });
const matTarget = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xff2d2d, emissiveIntensity: 0.9, roughness: 0.35 });
const matTargetHit = new THREE.MeshStandardMaterial({ color: 0xfbbf24, emissive: 0xffb020, emissiveIntensity: 1.4, roughness: 0.3 });
const matTrail = new THREE.MeshStandardMaterial({ color: 0xf59e0b, emissive: 0xf59e0b, emissiveIntensity: 0.9, roughness: 0.4 });
const matGhost = new THREE.MeshStandardMaterial({ color: 0x94a3b8, transparent: true, opacity: 0.22, depthWrite: false });

// ------------------------------------------------------------------ scene objects
const unitCyl = new THREE.CylinderGeometry(1, 1, 1, 28, 1, false);
const unitSphere = new THREE.SphereGeometry(1, 28, 20);

class ArmView {
  // Draws the arm either from frame origins (analytic backend: cylinders between joint
  // frames) or from the real collision capsules of the MuJoCo model (frame.capsules).
  constructor(linkRadius, ghost = false) {
    this.group = new THREE.Group();
    this.ghost = ghost;
    this.r = linkRadius;
    this.links = []; this.joints = []; this.caps = [];
    this.taper = [1.25, 1.0, 0.92, 0.8, 0.72, 0.66, 0.6, 0.55];
    if (!ghost) {
      const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.13, 0.02, 48), matBase);
      plate.position.y = 0.01; plate.castShadow = true; plate.receiveShadow = true; this.group.add(plate);
    }
    this._up = new THREE.Vector3(0, 1, 0);
  }
  _ensure(nLinks, nJoints) {
    while (this.links.length < nLinks) {
      const i = this.links.length;
      const m = new THREE.Mesh(unitCyl, this.ghost ? matGhost : (i === 0 ? matBase : matLink));
      m.castShadow = !this.ghost; m.receiveShadow = !this.ghost; m.userData.taper = this.taper[Math.min(i, this.taper.length - 1)];
      this.group.add(m); this.links.push(m);
    }
    while (this.joints.length < nJoints) {
      const s = new THREE.Mesh(unitSphere, this.ghost ? matGhost : matJoint);
      s.castShadow = !this.ghost; this.group.add(s); this.joints.push(s);
    }
  }
  _ensureCaps(n) {
    while (this.caps.length < n) {
      const g = new THREE.Group();
      const body = new THREE.Mesh(unitCyl, this.ghost ? matGhost : matLink);
      const c1 = new THREE.Mesh(unitSphere, this.ghost ? matGhost : matLink);
      const c2 = new THREE.Mesh(unitSphere, this.ghost ? matGhost : matLink);
      [body, c1, c2].forEach((m) => { m.castShadow = !this.ghost; m.receiveShadow = !this.ghost; g.add(m); });
      this.group.add(g); this.caps.push({ g, body, c1, c2 });
    }
  }
  setOrigins(origins, hit = false, capsules = null) {
    const pts = origins.map(toV);
    const n = pts.length - 1;
    this._ensure(n, pts.length);
    const useCaps = Array.isArray(capsules) && capsules.length > 0;
    for (let i = 0; i < this.links.length; i++) {
      const m = this.links[i];
      m.visible = i < n && !useCaps;
      if (!m.visible) continue;
      const a = pts[i], b = pts[i + 1];
      const dir = b.clone().sub(a); const len = Math.max(dir.length(), 1e-4);
      const rr = this.r * m.userData.taper;
      m.position.copy(a).addScaledVector(dir, 0.5);
      m.quaternion.setFromUnitVectors(this._up, dir.normalize());
      m.scale.set(rr, len, rr);
      if (!this.ghost && i > 0) m.material = hit ? matLinkHit : matLink;
    }
    for (let i = 0; i < this.joints.length; i++) {
      const j = this.joints[i];
      j.visible = i < pts.length;
      if (!j.visible) continue;
      const tp = this.taper[Math.min(i, this.taper.length - 1)];
      const rr = (i === 0 ? 1.3 : i === pts.length - 1 ? 0.55 : useCaps ? 0.5 : 1.12) * this.r * tp;
      j.position.copy(pts[i]); j.scale.setScalar(rr);
      if (!this.ghost) j.material = i === pts.length - 1 ? matEE : matJoint;
    }
    if (useCaps) {
      this._ensureCaps(capsules.length);
      capsules.forEach((c, k) => {
        const it = this.caps[k]; it.g.visible = true;
        const a = toV(c.a), b = toV(c.b); const dir = b.clone().sub(a); const len = dir.length();
        it.body.position.copy(a).addScaledVector(dir, 0.5);
        it.body.quaternion.setFromUnitVectors(this._up, len > 1e-6 ? dir.clone().normalize() : this._up);
        it.body.scale.set(c.r, Math.max(len, 1e-4), c.r);
        it.c1.position.copy(a); it.c1.scale.setScalar(c.r);
        it.c2.position.copy(b); it.c2.scale.setScalar(c.r);
        if (!this.ghost) { const mat = hit && c.link > 0 ? matLinkHit : (c.link === 0 ? matBase : matLink); it.body.material = mat; it.c1.material = mat; it.c2.material = mat; }
      });
      for (let k = capsules.length; k < this.caps.length; k++) this.caps[k].g.visible = false;
    } else {
      this.caps.forEach((it) => (it.g.visible = false));
    }
  }
}

class ObstacleView {
  constructor() { this.group = new THREE.Group(); this.items = []; this._up = new THREE.Vector3(0, 1, 0); }
  set(sceneData) {
    this.group.clear(); this.items = [];
    const rDefault = sceneData.obs_radius;
    sceneData.obstacles.forEach((o, idx) => {
      const mat = idx === 0 && sceneData.name === 'fruit_tree' ? matTrunk : matObs;
      const body = new THREE.Mesh(unitCyl, mat); const c1 = new THREE.Mesh(unitSphere, mat); const c2 = new THREE.Mesh(unitSphere, mat);
      [body, c1, c2].forEach((m) => { m.castShadow = true; m.receiveShadow = true; this.group.add(m); });
      this.items.push({ body, c1, c2, r: o.r ?? rDefault });
    });
    this.update(sceneData.obstacles);
  }
  update(obstacles) {
    obstacles.forEach((o, idx) => {
      const it = this.items[idx]; if (!it) return;
      const a = toV(o.p1), b = toV(o.p2); const dir = b.clone().sub(a); const len = dir.length();
      it.body.position.copy(a).addScaledVector(dir, 0.5);
      it.body.quaternion.setFromUnitVectors(this._up, len > 1e-6 ? dir.clone().normalize() : this._up);
      it.body.scale.set(it.r, Math.max(len, 1e-4), it.r);
      it.c1.position.copy(a); it.c1.scale.setScalar(it.r);
      it.c2.position.copy(b); it.c2.scale.setScalar(it.r);
    });
  }
}

class TargetView {
  constructor() {
    this.group = new THREE.Group();
    this.sphere = new THREE.Mesh(unitSphere, matTarget); this.sphere.castShadow = true; this.group.add(this.sphere);
    this.halo = new THREE.Mesh(new THREE.RingGeometry(1.0, 1.08, 64), new THREE.MeshBasicMaterial({ color: 0xff6b6b, transparent: true, opacity: 0.5, side: THREE.DoubleSide }));
    this.group.add(this.halo);
    this.t = 0;
  }
  set(target, radius) { this.group.position.copy(toV(target)); this.sphere.scale.setScalar(radius); this.halo.scale.setScalar(radius * 1.6); }
  setPosition(target) { this.group.position.copy(toV(target)); }
  update(dt, reached) {
    this.t += dt;
    this.sphere.material = reached ? matTargetHit : matTarget;
    this.halo.lookAt(camera.position);
    const s = reached ? 1.6 + 0.5 * (0.5 + 0.5 * Math.sin(this.t * 6)) : 1.6 + 0.15 * Math.sin(this.t * 2.5);
    this.halo.scale.setScalar(this.sphere.scale.x * s);
    this.halo.material.color.set(reached ? 0xfbbf24 : 0xff6b6b);
  }
}

class TrailView {
  constructor() { this.mesh = null; this.group = new THREE.Group(); }
  set(points) {
    if (this.mesh) { this.group.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh = null; }
    if (points.length < 2) return;
    const curve = new THREE.CatmullRomCurve3(points.map(toV), false, 'centripetal', 0.2);
    const geo = new THREE.TubeGeometry(curve, Math.max(8, points.length * 3), 0.0045, 10, false);
    this.mesh = new THREE.Mesh(geo, matTrail); this.group.add(this.mesh);
  }
}

class SlView {
  // One emissive shaft + cone per monitored link: P' (on the link axis) -> Q' (on the obstacle axis).
  constructor() { this.group = new THREE.Group(); this.items = []; this._up = new THREE.Vector3(0, 1, 0); }
  _make() {
    const mat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.7, roughness: 0.5 });
    const shaft = new THREE.Mesh(unitCyl, mat); const head = new THREE.Mesh(new THREE.ConeGeometry(1, 1, 16), mat);
    const tip = new THREE.Mesh(unitSphere, mat);
    const g = new THREE.Group(); g.add(shaft, head, tip); this.group.add(g);
    const it = { g, shaft, head, tip, mat }; this.items.push(it); return it;
  }
  set(sl) {
    while (this.items.length < sl.length) this._make();
    this.items.forEach((it, i) => {
      if (i >= sl.length) { it.g.visible = false; return; }
      const p = toV(sl[i].p), q = toV(sl[i].q); const d = q.clone().sub(p); const len = d.length();
      it.g.visible = len > 1e-3;
      const dir = d.clone().normalize(); const headLen = Math.min(0.03, len * 0.35), r = 0.0035;
      it.shaft.position.copy(p).addScaledVector(dir, (len - headLen) / 2); it.shaft.quaternion.setFromUnitVectors(this._up, dir); it.shaft.scale.set(r, Math.max(len - headLen, 1e-4), r);
      it.head.position.copy(p).addScaledVector(dir, len - headLen / 2); it.head.quaternion.setFromUnitVectors(this._up, dir); it.head.scale.set(r * 2.6, headLen, r * 2.6);
      it.tip.position.copy(p); it.tip.scale.setScalar(r * 1.8);
      const c = THREE.MathUtils.clamp(sl[i].d / 0.25, 0, 1); // surface clearance -> red (0) .. green (25 cm)
      it.mat.color.setHSL(c * 0.36, 0.95, 0.55); it.mat.emissive.copy(it.mat.color);
    });
  }
}

const armView = new ArmView(0.04); scene.add(armView.group);
const ghostView = new ArmView(0.04, true); ghostView.group.visible = false; scene.add(ghostView.group);
const obsView = new ObstacleView(); scene.add(obsView.group);
const targetView = new TargetView(); scene.add(targetView.group);
const trailView = new TrailView(); scene.add(trailView.group);
const slView = new SlView(); scene.add(slView.group);

// ------------------------------------------------------------------ state
const S = {
  manifest: null, ckptIdx: 0, ckptData: null, cache: new Map(), baseline: null,
  sceneIdx: 0, frame: 0, playing: false, speed: 1, acc: 0, frac: 0,
  opts: { sl: true, trail: true, ghost: false, rotate: false, loop: true, follow: true },
};
const episode = () => S.ckptData?.episodes?.[S.sceneIdx];

async function fetchJSON(path) { const r = await fetch(path + (path.includes('?') ? '&' : '?') + 't=' + Date.now(), { cache: 'no-store' }); if (!r.ok) throw new Error(`${r.status} ${path}`); return r.json(); }

async function loadManifest(initial = false) {
  let m;
  try { m = await fetchJSON(DATA + 'manifest.json'); } catch (e) { setStatus('bad', `no data at ${RUN}/viewer (run scripts/export_viewer.py)`); return; }
  const prevN = S.manifest?.checkpoints?.length || 0;
  S.manifest = m;
  $('run-name').textContent = `${m.run} · ${m.backend === 'mujoco' ? 'MuJoCo UR5e' : 'analytic arm'}`;
  if (m.reach) ring.scale.setScalar(m.reach / 0.6075);
  $('live-badge').classList.toggle('hidden', !!m.done);
  const n = m.checkpoints.length;
  $('ckpt-slider').max = Math.max(0, n - 1);
  renderDots();
  if (initial) {
    S.ckptIdx = n - 1; await loadCheckpoint(S.ckptIdx);
    if (m.backend === 'mujoco') { camera.position.set(-0.9, 1.4, 2.4); controls.target.set(0.45, 0.4, 0.0); }
  }
  else if (n > prevN && S.opts.follow) { S.ckptIdx = n - 1; await loadCheckpoint(S.ckptIdx); toast(`new checkpoint: round ${m.checkpoints[n - 1].episode}`); }
  drawCurve();
  setStatus(m.done ? 'ok' : '', m.done ? `${n} checkpoints` : `training… ${n} checkpoints`);
  if (!m.baseline_checked && m.checkpoints[0]?.episode === 0) S.baseline = await getCkpt(m.checkpoints[0].file);
}

async function getCkpt(file) { if (!S.cache.has(file)) S.cache.set(file, await fetchJSON(DATA + file)); return S.cache.get(file); }

async function loadCheckpoint(idx, keepFrame = false) {
  const c = S.manifest.checkpoints[idx]; if (!c) return;
  S.ckptIdx = idx; $('ckpt-slider').value = idx;
  S.ckptData = await getCkpt(c.file);
  if (!S.baseline && S.manifest.checkpoints[0]?.episode === 0) S.baseline = await getCkpt(S.manifest.checkpoints[0].file);
  $('ckpt-label').textContent = `round ${c.episode}  ·  ${Math.round(c.success_rate * 100)}% ✓`;
  if (S.sceneIdx >= S.ckptData.episodes.length) S.sceneIdx = 0;
  renderDots(); renderTabs(); drawCurve();
  setScene(S.sceneIdx, keepFrame);
}

function setScene(i, keepFrame = false) {
  S.sceneIdx = i; const ep = episode(); if (!ep) return;
  obsView.set(ep.scene); targetView.set(ep.scene.target, ep.scene.target_radius);
  if (!keepFrame) { S.frame = 0; S.frac = 0; }
  S.frame = Math.min(S.frame, ep.frames.length - 1);
  $('frame-slider').max = ep.frames.length - 1;
  renderTabs(); applyFrame(); updateHud();
  $('scene-summary').textContent = `${ep.scene.name || 'scene'}${ep.scene.dynamic ? ' · moving' : ''} · ${ep.steps} steps · L=${fmt(ep.path_length, 2)} m`;
}

function applyFrame() {
  const ep = episode(); if (!ep) return;
  const k = S.frame, f = ep.frames[k], g = ep.frames[Math.min(k + 1, ep.frames.length - 1)];
  const t = S.playing ? S.frac : 0;
  const origins = f.origins.map((o, i) => o.map((v, j) => lerp(v, g.origins[i][j], t)));
  let caps = null;
  if (f.capsules) caps = f.capsules.map((c, i) => ({ a: c.a.map((v, j) => lerp(v, g.capsules[i].a[j], t)), b: c.b.map((v, j) => lerp(v, g.capsules[i].b[j], t)), r: c.r, link: c.link }));
  armView.setOrigins(origins, f.collision || (t > 0.5 && g.collision), caps);
  if (f.obstacles) obsView.update(f.obstacles.map((o, i) => ({ p1: o.p1.map((v, j) => lerp(v, g.obstacles[i].p1[j], t)), p2: o.p2.map((v, j) => lerp(v, g.obstacles[i].p2[j], t)) })));
  if (f.target) targetView.setPosition(f.target.map((v, j) => lerp(v, g.target[j], t)));
  trailView.group.visible = S.opts.trail;
  if (S.opts.trail) trailView.set(ep.frames.slice(0, k + 1).map((fr) => fr.ee).concat([origins[6]]));
  slView.group.visible = S.opts.sl; if (S.opts.sl) slView.set(f.sl);
  // ghost: untrained policy on the same scene, same step index
  const base = S.baseline?.episodes?.[S.sceneIdx];
  ghostView.group.visible = S.opts.ghost && !!base && S.manifest.checkpoints[S.ckptIdx].episode !== 0;
  if (ghostView.group.visible) { const bf = base.frames[Math.min(k, base.frames.length - 1)]; ghostView.setOrigins(bf.origins, false, bf.capsules || null); }
  $('frame-slider').value = k;
}

function updateHud() {
  const ep = episode(); if (!ep) return; const f = ep.frames[S.frame];
  $('hud-step').textContent = `${S.frame} / ${ep.frames.length - 1}`;
  const d = $('hud-dist'); d.textContent = `${(f.distance * 100).toFixed(1)} cm`; d.className = 'v mono ' + (f.distance < 0.03 ? 'ok' : f.distance < 0.1 ? 'warn' : '');
  $('hud-reward').textContent = fmt(f.reward, 2);
  $('hud-return').textContent = fmt(f.return, 1);
  const minc = Math.min(...f.sl.map((s) => s.d)); const c = $('hud-clear'); c.textContent = `${(minc * 100).toFixed(1)} cm`; c.className = 'v mono ' + (minc < 0 ? 'bad' : minc < 0.03 ? 'warn' : '');
  const o = $('hud-outcome');
  if (f.collision) { o.textContent = 'collision'; o.className = 'v bad'; }
  else if (f.reached) { o.textContent = 'target reached'; o.className = 'v ok'; }
  else if (S.frame === ep.frames.length - 1) { o.textContent = 'timeout'; o.className = 'v warn'; }
  else { o.textContent = 'moving'; o.className = 'v'; }
  $('joints').innerHTML = f.q.map((q, i) => `<div>θ${i + 1}<b>${(q * 180 / Math.PI).toFixed(0)}°</b></div>`).join('');
}

function setStatus(cls, text) { const p = $('status-pill'); p.className = 'status-pill ' + cls; $('status-text').textContent = text; }

function renderDots() {
  const box = $('ckpt-dots'); box.innerHTML = '';
  S.manifest.checkpoints.forEach((c, i) => {
    const d = document.createElement('i');
    const h = c.success_rate * 0.36; d.style.background = `hsl(${h * 360} 80% ${35 + c.success_rate * 20}%)`;
    d.title = `round ${c.episode}: ${Math.round(c.success_rate * 100)}% success, return ${c.mean_return.toFixed(1)}`;
    if (i === S.ckptIdx) d.classList.add('cur');
    d.onclick = () => loadCheckpoint(i, true); box.appendChild(d);
  });
}

function renderTabs() {
  const box = $('scene-tabs'); box.innerHTML = '';
  (S.ckptData?.episodes || []).forEach((ep, i) => {
    const b = document.createElement('button'); b.className = i === S.sceneIdx ? 'cur' : '';
    b.innerHTML = `<span>scene ${i + 1}</span><span class="res ${ep.success ? 'ok' : 'bad'}">${ep.success ? '✓ ' + ep.steps : ep.collided ? '✗ hit' : '✗ miss'}</span>`;
    b.onclick = () => setScene(i); box.appendChild(b);
  });
}

// ------------------------------------------------------------------ training curve
const curveCanvas = $('curve');
function drawCurve() {
  const m = S.manifest; if (!m) return;
  const dpr = Math.min(devicePixelRatio, 2); const W = curveCanvas.clientWidth, H = 150;
  curveCanvas.width = W * dpr; curveCanvas.height = H * dpr;
  const ctx = curveCanvas.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const pad = { l: 36, r: 30, t: 10, b: 20 };
  const curve = m.curve || []; const cks = m.checkpoints;
  const xmax = Math.max(curve.at(-1)?.episode || 0, cks.at(-1)?.episode || 1, 1);
  const rets = curve.map((c) => c.return); const ymin = Math.min(...rets, -1), ymax = Math.max(...rets, 0);
  const X = (e) => pad.l + (e / xmax) * (W - pad.l - pad.r);
  const Y = (r) => pad.t + (1 - (r - ymin) / (ymax - ymin || 1)) * (H - pad.t - pad.b);
  const Ys = (s) => pad.t + (1 - s) * (H - pad.t - pad.b);
  ctx.strokeStyle = 'rgba(148,163,184,.15)'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const y = pad.t + (i / 4) * (H - pad.t - pad.b); ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke(); }
  ctx.fillStyle = '#8b96ab'; ctx.font = '10px JetBrains Mono, monospace'; ctx.textAlign = 'right';
  ctx.fillText(ymax.toFixed(0), pad.l - 4, pad.t + 4); ctx.fillText(ymin.toFixed(0), pad.l - 4, H - pad.b);
  ctx.textAlign = 'left'; ctx.fillText('100%', W - pad.r + 3, pad.t + 4); ctx.fillText('0%', W - pad.r + 3, H - pad.b);
  ctx.textAlign = 'center'; ctx.fillText('0', pad.l, H - 5); ctx.fillText(String(xmax), W - pad.r, H - 5);
  if (curve.length) {
    // success area
    ctx.beginPath(); ctx.moveTo(X(curve[0].episode), Ys(0));
    curve.forEach((c) => ctx.lineTo(X(c.episode), Ys(c.success))); ctx.lineTo(X(curve.at(-1).episode), Ys(0)); ctx.closePath();
    const gr = ctx.createLinearGradient(0, pad.t, 0, H - pad.b); gr.addColorStop(0, 'rgba(52,211,153,.35)'); gr.addColorStop(1, 'rgba(52,211,153,0)');
    ctx.fillStyle = gr; ctx.fill();
    // return line
    ctx.beginPath(); curve.forEach((c, i) => (i ? ctx.lineTo(X(c.episode), Y(c.return)) : ctx.moveTo(X(c.episode), Y(c.return))));
    ctx.strokeStyle = '#60a5fa'; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
  }
  // checkpoint markers
  cks.forEach((c, i) => {
    const x = X(c.episode); ctx.beginPath(); ctx.arc(x, Ys(c.success_rate), i === S.ckptIdx ? 5 : 2.5, 0, Math.PI * 2);
    ctx.fillStyle = i === S.ckptIdx ? '#fff' : 'rgba(255,255,255,.55)'; ctx.fill();
    if (i === S.ckptIdx) { ctx.strokeStyle = 'rgba(255,255,255,.35)'; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, H - pad.b); ctx.stroke(); ctx.setLineDash([]); }
  });
}
curveCanvas.addEventListener('click', (e) => {
  const m = S.manifest; if (!m) return; const rect = curveCanvas.getBoundingClientRect(); const x = e.clientX - rect.left;
  const W = rect.width, pad = { l: 36, r: 30 }; const xmax = Math.max(m.curve.at(-1)?.episode || 0, m.checkpoints.at(-1)?.episode || 1, 1);
  const ep = ((x - pad.l) / (W - pad.l - pad.r)) * xmax;
  let best = 0; m.checkpoints.forEach((c, i) => { if (Math.abs(c.episode - ep) < Math.abs(m.checkpoints[best].episode - ep)) best = i; });
  loadCheckpoint(best, true);
});

// ------------------------------------------------------------------ playback
function setPlaying(p) { S.playing = p; $('btn-play').textContent = p ? '❚❚' : '▶'; if (!p) { S.frac = 0; applyFrame(); } }
function stepFrame(delta) { const ep = episode(); if (!ep) return; S.frame = THREE.MathUtils.clamp(S.frame + delta, 0, ep.frames.length - 1); S.frac = 0; applyFrame(); updateHud(); }
function gotoFrame(k) { const ep = episode(); if (!ep) return; S.frame = THREE.MathUtils.clamp(k, 0, ep.frames.length - 1); S.frac = 0; applyFrame(); updateHud(); }

$('btn-play').onclick = () => { const ep = episode(); if (ep && S.frame >= ep.frames.length - 1) S.frame = 0; setPlaying(!S.playing); };
$('btn-prev').onclick = () => { setPlaying(false); stepFrame(-1); };
$('btn-next').onclick = () => { setPlaying(false); stepFrame(1); };
$('btn-first').onclick = () => { setPlaying(false); gotoFrame(0); };
$('btn-last').onclick = () => { setPlaying(false); gotoFrame(1e9); };
$('speed').onchange = (e) => (S.speed = parseFloat(e.target.value));
$('frame-slider').oninput = (e) => { setPlaying(false); gotoFrame(parseInt(e.target.value, 10)); };
$('ckpt-slider').oninput = (e) => loadCheckpoint(parseInt(e.target.value, 10), true);
for (const [id, key] of [['opt-sl', 'sl'], ['opt-trail', 'trail'], ['opt-ghost', 'ghost'], ['opt-rotate', 'rotate'], ['opt-loop', 'loop'], ['follow-latest', 'follow']]) {
  $(id).onchange = (e) => { S.opts[key] = e.target.checked; controls.autoRotate = S.opts.rotate; applyFrame(); };
}
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); $('btn-play').click(); }
  else if (e.key === 'ArrowRight') $('btn-next').click();
  else if (e.key === 'ArrowLeft') $('btn-prev').click();
  else if (e.key === 'Home') $('btn-first').click();
  else if (e.key === 'End') $('btn-last').click();
  else if (e.key === ']') loadCheckpoint(Math.min(S.ckptIdx + 1, S.manifest.checkpoints.length - 1), true);
  else if (e.key === '[') loadCheckpoint(Math.max(S.ckptIdx - 1, 0), true);
  else if (/^[1-9]$/.test(e.key)) { const i = parseInt(e.key, 10) - 1; if (S.ckptData?.episodes[i]) setScene(i); }
});

// ------------------------------------------------------------------ loop
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.1);
  const ep = episode();
  if (S.playing && ep) {
    S.frac += dt * 8 * S.speed; // 8 sim steps per second at 1x
    while (S.frac >= 1) {
      S.frac -= 1; S.frame += 1;
      if (S.frame >= ep.frames.length - 1) {
        S.frame = ep.frames.length - 1;
        if (S.opts.loop) { S.frame = 0; } else { setPlaying(false); break; }
      }
      updateHud();
    }
    applyFrame();
  }
  targetView.update(dt, !!ep?.frames[S.frame]?.reached);
  controls.update();
  composer.render();
}
animate();

loadManifest(true).then(() => { setPlaying(true); });
setInterval(() => { if (S.manifest && !S.manifest.done) loadManifest(false); }, 4000);
