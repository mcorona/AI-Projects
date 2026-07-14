import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

/* ---------------------------------------------------------------------
   Datos de los planetas (unidades escaladas para visualizacion, no a
   escala real: si lo fuera, los planetas serian invisibles a simple
   vista frente a las distancias orbitales).
--------------------------------------------------------------------- */
const PLANETS = [
  {
    key: "mercury", name: "Mercurio", color: 0x9c9284, size: 0.38,
    distance: 9, speed: 4.15, rotSpeed: 0.006, tilt: 0.03,
    desc: "El planeta mas pequeno y el mas cercano al Sol. Un dia dura mas que su ano.",
    stats: { "Distancia al Sol": "57.9 M km", "Diametro": "4,879 km", "Ano": "88 dias", "Lunas": "0" },
  },
  {
    key: "venus", name: "Venus", color: 0xe8c88a, size: 0.92,
    distance: 12.5, speed: 1.62, rotSpeed: -0.0025, tilt: 3.1,
    desc: "El planeta mas caliente del sistema solar por su densa atmosfera de CO2. Gira al reves.",
    stats: { "Distancia al Sol": "108.2 M km", "Diametro": "12,104 km", "Ano": "225 dias", "Lunas": "0" },
  },
  {
    key: "earth", name: "Tierra", color: 0x2e6fd6, size: 0.97,
    distance: 16.5, speed: 1, rotSpeed: 0.03, tilt: 0.41,
    desc: "Nuestro hogar. El unico planeta conocido con vida y agua liquida en su superficie.",
    stats: { "Distancia al Sol": "149.6 M km", "Diametro": "12,742 km", "Ano": "365 dias", "Lunas": "1" },
    moons: [{ name: "Luna", color: 0xbfbfbf, size: 0.26, distance: 1.7, speed: 8 }],
  },
  {
    key: "mars", name: "Marte", color: 0xc1592f, size: 0.53,
    distance: 21, speed: 0.53, rotSpeed: 0.029, tilt: 0.44,
    desc: "El planeta rojo, cubierto de oxido de hierro. Alberga el volcan mas grande conocido.",
    stats: { "Distancia al Sol": "227.9 M km", "Diametro": "6,779 km", "Ano": "687 dias", "Lunas": "2" },
  },
  {
    key: "jupiter", name: "Jupiter", color: 0xd8b48c, size: 2.6,
    distance: 29, speed: 0.084, rotSpeed: 0.07, tilt: 0.05,
    desc: "El gigante gaseoso mas grande. Su Gran Mancha Roja es una tormenta mayor que la Tierra.",
    stats: { "Distancia al Sol": "778.5 M km", "Diametro": "139,820 km", "Ano": "12 anos", "Lunas": "95" },
  },
  {
    key: "saturn", name: "Saturno", color: 0xe3d0a3, size: 2.2,
    distance: 37, speed: 0.034, rotSpeed: 0.065, tilt: 0.47,
    desc: "Famoso por su espectacular sistema de anillos formados por hielo y roca.",
    stats: { "Distancia al Sol": "1,434 M km", "Diametro": "116,460 km", "Ano": "29 anos", "Lunas": "146" },
    ring: { inner: 2.6, outer: 4.3, color: 0xcbb98f },
  },
  {
    key: "uranus", name: "Urano", color: 0x9fd8e0, size: 1.5,
    distance: 44, speed: 0.012, rotSpeed: 0.045, tilt: 1.71,
    desc: "Gigante de hielo que gira practicamente de lado sobre su eje.",
    stats: { "Distancia al Sol": "2,871 M km", "Diametro": "50,724 km", "Ano": "84 anos", "Lunas": "27" },
    ring: { inner: 1.8, outer: 2.4, color: 0x8fb8c0 },
  },
  {
    key: "neptune", name: "Neptuno", color: 0x3a5fcd, size: 1.45,
    distance: 50, speed: 0.006, rotSpeed: 0.045, tilt: 0.49,
    desc: "El planeta mas ventoso, con rachas de hasta 2,100 km/h. Toma 165 anos en orbitar el Sol.",
    stats: { "Distancia al Sol": "4,495 M km", "Diametro": "49,244 km", "Ano": "165 anos", "Lunas": "16" },
  },
];

const SUN = {
  key: "sun", name: "Sol", color: 0xffcc55, size: 5,
  desc: "La estrella del sistema solar. Concentra el 99.8% de toda su masa.",
  stats: { "Diametro": "1,391,400 km", "Temperatura": "5,500 °C", "Edad": "4,600 M anos", "Tipo": "Enana amarilla" },
};

/* ---------------------------------------------------------------------
   Escena
--------------------------------------------------------------------- */
const app = document.getElementById("app");
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 4000);
camera.position.set(0, 34, 68);

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
app.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 4;
controls.maxDistance = 220;
controls.target.set(0, 0, 0);

// Estrellas de fondo
function buildStars() {
  const count = 4000;
  const positions = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const r = 400 + Math.random() * 900;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);
    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.cos(phi);
    positions[i * 3 + 2] = r * Math.sin(phi) * Math.sin(theta);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({ color: 0xffffff, size: 1.1, sizeAttenuation: true });
  scene.add(new THREE.Points(geo, mat));
}
buildStars();

// Luz del Sol
const sunLight = new THREE.PointLight(0xffffff, 4.2, 0, 0.0008);
sunLight.position.set(0, 0, 0);
scene.add(sunLight);
scene.add(new THREE.AmbientLight(0x394268, 0.55));

// Sol
const sunGeo = new THREE.SphereGeometry(SUN.size, 48, 48);
const sunMat = new THREE.MeshBasicMaterial({ color: SUN.color });
const sunMesh = new THREE.Mesh(sunGeo, sunMat);
sunMesh.userData.info = SUN;
scene.add(sunMesh);

// Resplandor del sol (sprite aditivo)
function buildGlowTexture() {
  const size = 256;
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(255,220,150,0.9)");
  g.addColorStop(0.4, "rgba(255,180,80,0.35)");
  g.addColorStop(1, "rgba(255,150,50,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  return new THREE.CanvasTexture(c);
}
const glowSprite = new THREE.Sprite(new THREE.SpriteMaterial({
  map: buildGlowTexture(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false,
}));
glowSprite.scale.set(SUN.size * 6, SUN.size * 6, 1);
scene.add(glowSprite);

// Grupo de orbitas (para poder ocultarlas)
const orbitsGroup = new THREE.Group();
scene.add(orbitsGroup);

function buildOrbitLine(radius) {
  const points = [];
  const segments = 128;
  for (let i = 0; i <= segments; i++) {
    const a = (i / segments) * Math.PI * 2;
    points.push(new THREE.Vector3(Math.cos(a) * radius, 0, Math.sin(a) * radius));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(points);
  const mat = new THREE.LineBasicMaterial({ color: 0x4a5480, transparent: true, opacity: 0.45 });
  return new THREE.LineLoop(geo, mat);
}

// Planetas
const planetObjects = [];
const labelEls = new Map();
const labelsLayer = document.createElement("div");
labelsLayer.style.cssText = "position:fixed;inset:0;pointer-events:none;z-index:10;";
document.body.appendChild(labelsLayer);
let labelsVisible = false;

for (let pi = 0; pi < PLANETS.length; pi++) {
  const p = PLANETS[pi];
  const pivot = new THREE.Object3D();
  // Angulo de oro para repartir los planetas y evitar que alguno quede
  // alineado justo con la camara al cargar (se veria gigante/recortado).
  pivot.rotation.y = pi * 2.39996;
  scene.add(pivot);

  orbitsGroup.add(buildOrbitLine(p.distance));

  const geo = new THREE.SphereGeometry(p.size, 40, 40);
  const mat = new THREE.MeshStandardMaterial({ color: p.color, roughness: 0.85, metalness: 0.05 });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.x = p.distance;
  mesh.rotation.z = p.tilt;
  mesh.userData.info = p;
  pivot.add(mesh);

  if (p.ring) {
    const ringGeo = new THREE.RingGeometry(p.ring.inner, p.ring.outer, 64);
    // UVs radiales para un degradado mas realista
    const pos = ringGeo.attributes.position;
    const v3 = new THREE.Vector3();
    for (let i = 0; i < pos.count; i++) {
      v3.fromBufferAttribute(pos, i);
      ringGeo.attributes.uv.setXY(i, v3.length() < (p.ring.inner + p.ring.outer) / 2 ? 0 : 1, 1);
    }
    const ringMat = new THREE.MeshBasicMaterial({
      color: p.ring.color, side: THREE.DoubleSide, transparent: true, opacity: 0.75,
    });
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    ringMesh.rotation.x = Math.PI / 2 - 0.35;
    mesh.add(ringMesh);
  }

  // Lunas
  const moonMeshes = [];
  if (p.moons) {
    for (const m of p.moons) {
      const moonPivot = new THREE.Object3D();
      mesh.add(moonPivot);
      const moonMesh = new THREE.Mesh(
        new THREE.SphereGeometry(m.size, 16, 16),
        new THREE.MeshStandardMaterial({ color: m.color, roughness: 0.9 })
      );
      moonMesh.position.x = m.distance;
      moonPivot.add(moonMesh);
      moonMeshes.push({ pivot: moonPivot, speed: m.speed });
    }
  }

  const label = document.createElement("div");
  label.textContent = p.name;
  label.style.cssText = "position:absolute;transform:translate(-50%,-140%);font-size:11px;color:#dfe3f5;background:rgba(8,10,24,0.55);padding:2px 7px;border-radius:8px;white-space:nowrap;border:1px solid rgba(255,255,255,0.08);opacity:0;transition:opacity .2s;";
  labelsLayer.appendChild(label);
  labelEls.set(p.key, label);

  planetObjects.push({ ...p, pivot, mesh, moonMeshes, angle: pivot.rotation.y });
}

/* ---------------------------------------------------------------------
   Panel lateral con la lista de planetas (acceso rapido en movil)
--------------------------------------------------------------------- */
const listEl = document.getElementById("planetList");
function addListItem(key, name, color) {
  const item = document.createElement("div");
  item.className = "item";
  item.innerHTML = `<div class="sw" style="background:#${color.toString(16).padStart(6, "0")}"></div><span>${name}</span>`;
  item.addEventListener("click", () => focusOn(key));
  listEl.appendChild(item);
}
addListItem("sun", SUN.name, SUN.color);
for (const p of PLANETS) addListItem(p.key, p.name, p.color);

/* ---------------------------------------------------------------------
   Interaccion: tocar un planeta -> panel de info + enfoque de camara
--------------------------------------------------------------------- */
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let selectedKey = null;

const panel = document.getElementById("panel");
const panelBackdrop = document.getElementById("panelBackdrop");
const panelName = document.getElementById("panelName");
const panelDesc = document.getElementById("panelDesc");
const panelSwatch = document.getElementById("panelSwatch");
const panelStats = document.getElementById("panelStats");
const hint = document.getElementById("hint");

function showPanel(info) {
  panelName.textContent = info.name;
  panelDesc.textContent = info.desc;
  panelSwatch.style.background = "#" + info.color.toString(16).padStart(6, "0");
  panelStats.innerHTML = "";
  for (const [k, v] of Object.entries(info.stats)) {
    const el = document.createElement("div");
    el.className = "stat";
    el.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div>`;
    panelStats.appendChild(el);
  }
  panel.classList.add("open");
  panelBackdrop.classList.add("open");
  hint.classList.add("hidden");
}
function closePanel() {
  panel.classList.remove("open");
  panelBackdrop.classList.remove("open");
  selectedKey = null;
}
document.getElementById("panelClose").addEventListener("click", closePanel);
panelBackdrop.addEventListener("click", closePanel);

function focusOn(key) {
  selectedKey = key;
  if (key === "sun") {
    showPanel(SUN);
    animateCameraTo(new THREE.Vector3(0, 8, 22), new THREE.Vector3(0, 0, 0));
    return;
  }
  const obj = planetObjects.find((p) => p.key === key);
  if (!obj) return;
  showPanel(obj);
  const worldPos = new THREE.Vector3();
  obj.mesh.getWorldPosition(worldPos);
  const dist = Math.max(obj.size * 6, 3.5);
  const dir = worldPos.clone().normalize();
  const camPos = worldPos.clone().add(dir.multiplyScalar(dist)).add(new THREE.Vector3(0, dist * 0.5, 0));
  animateCameraTo(camPos, worldPos);
}

let camAnim = null;
function animateCameraTo(pos, target) {
  camAnim = { from: camera.position.clone(), to: pos, tFrom: controls.target.clone(), tTo: target, t: 0 };
  controls.enabled = false;
}

function getPointer(evt) {
  const rect = renderer.domElement.getBoundingClientRect();
  const x = evt.clientX ?? (evt.touches && evt.touches[0].clientX);
  const y = evt.clientY ?? (evt.touches && evt.touches[0].clientY);
  pointer.x = ((x - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((y - rect.top) / rect.height) * 2 + 1;
}

let downX = 0, downY = 0;
renderer.domElement.addEventListener("pointerdown", (e) => { downX = e.clientX; downY = e.clientY; });
renderer.domElement.addEventListener("pointerup", (e) => {
  const moved = Math.hypot(e.clientX - downX, e.clientY - downY);
  if (moved > 8) return; // fue un arrastre, no un tap
  getPointer(e);
  raycaster.setFromCamera(pointer, camera);
  const targets = [sunMesh, ...planetObjects.map((p) => p.mesh)];
  const hits = raycaster.intersectObjects(targets, false);
  if (hits.length) {
    const info = hits[0].object.userData.info;
    focusOn(info.key);
  }
});

/* ---------------------------------------------------------------------
   Controles de la barra inferior
--------------------------------------------------------------------- */
let paused = false;
let timeScale = 1;
const speedSteps = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16];
let speedIndex = 3;

const btnPause = document.getElementById("btnPause");
btnPause.addEventListener("click", () => {
  paused = !paused;
  btnPause.textContent = paused ? "▶️" : "⏸️";
});
const speedLabel = document.getElementById("speedLabel");
function updateSpeedLabel() { speedLabel.textContent = speedSteps[speedIndex] + "x"; }
document.getElementById("btnSlower").addEventListener("click", () => {
  speedIndex = Math.max(0, speedIndex - 1);
  updateSpeedLabel();
});
document.getElementById("btnFaster").addEventListener("click", () => {
  speedIndex = Math.min(speedSteps.length - 1, speedIndex + 1);
  updateSpeedLabel();
});
updateSpeedLabel();

let orbitsVisible = true;
document.getElementById("btnOrbits").addEventListener("click", () => {
  orbitsVisible = !orbitsVisible;
  orbitsGroup.visible = orbitsVisible;
});

document.getElementById("btnReset").addEventListener("click", () => {
  closePanel();
  animateCameraTo(new THREE.Vector3(0, 34, 68), new THREE.Vector3(0, 0, 0));
});

const btnLabels = document.getElementById("btnLabels");
btnLabels.addEventListener("click", () => {
  labelsVisible = !labelsVisible;
  for (const el of labelEls.values()) el.style.opacity = labelsVisible ? "1" : "0";
});

/* ---------------------------------------------------------------------
   Resize
--------------------------------------------------------------------- */
window.addEventListener("resize", () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

/* ---------------------------------------------------------------------
   Loop de animacion
--------------------------------------------------------------------- */
const tmpVec = new THREE.Vector3();
const clock = new THREE.Clock();

function updateLabels() {
  for (const p of planetObjects) {
    const el = labelEls.get(p.key);
    if (!el || el.style.opacity === "0") continue;
    p.mesh.getWorldPosition(tmpVec);
    tmpVec.project(camera);
    const behind = tmpVec.z > 1;
    el.style.display = behind ? "none" : "block";
    el.style.left = ((tmpVec.x * 0.5 + 0.5) * window.innerWidth) + "px";
    el.style.top = ((-tmpVec.y * 0.5 + 0.5) * window.innerHeight) + "px";
  }
}

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05);

  if (!paused) {
    const scale = speedSteps[speedIndex];
    for (const p of planetObjects) {
      p.pivot.rotation.y += dt * 0.25 * p.speed * scale;
      p.mesh.rotation.y += dt * p.rotSpeed * 20 * scale;
      for (const m of p.moonMeshes) m.pivot.rotation.y += dt * 0.25 * m.speed * scale;
    }
    sunMesh.rotation.y += dt * 0.02;
  }

  if (camAnim) {
    camAnim.t = Math.min(1, camAnim.t + dt * 1.4);
    const e = 1 - Math.pow(1 - camAnim.t, 3);
    camera.position.lerpVectors(camAnim.from, camAnim.to, e);
    controls.target.lerpVectors(camAnim.tFrom, camAnim.tTo, e);
    if (camAnim.t >= 1) { camAnim = null; controls.enabled = true; }
  }

  controls.update();
  if (labelsVisible) updateLabels();
  renderer.render(scene, camera);
}

/* ---------------------------------------------------------------------
   Arranque
--------------------------------------------------------------------- */
const loading = document.getElementById("loading");
loading.style.display = "none";
setTimeout(() => hint.classList.add("hidden"), 6000);
animate();

if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("sw.js").catch(() => {});
  });
}
