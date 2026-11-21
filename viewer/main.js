import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const stage = document.querySelector('#stage');
const status = document.querySelector('#status');
const clip = document.querySelector('#clip');
for (const button of document.querySelectorAll('[data-clip]')) {
  button.onclick = async () => {
    clip.src = `/model/${button.dataset.clip}`;
    try { await clip.play(); }
    catch (error) { if (error.name !== 'AbortError') { status.hidden = false; status.textContent = 'Could not play this voice clip.'; } }
  };
}

async function start() {
  const renderer = new THREE.WebGLRenderer({ canvas: stage.querySelector('canvas'), alpha: true, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(31, 1, .01, 40);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.minDistance = .17;
  controls.maxDistance = 9;
  const render = () => renderer.render(scene, camera);
  controls.addEventListener('change', render);
  function resize() {
    const { clientWidth: w, clientHeight: h } = stage;
    if (!w || !h) return;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
    render();
  }
  new ResizeObserver(resize).observe(stage);
  resize();
  scene.add(new THREE.HemisphereLight(0xfff7e9, 0xb0b6b5, 1.65));
  function light(color, intensity, x, y, z) {
    const lamp = new THREE.DirectionalLight(color, intensity);
    lamp.position.set(x, y, z);
    scene.add(lamp);
  }
  light(0xffe8cc, 1.75, -3, 5, 4);
  light(0xd3e9ff, 1.2, 2, 3, -2);
  const { scene: model } = await new GLTFLoader().loadAsync('/model/skin.glb');
  model.traverse(node => {
    if (!node.isMesh) return;
    // The Source 2 tear film and shadow overlay don't have glTF equivalents.
    if (['gman_eyemeniscus', 'gman_eyeshadow'].includes(node.material.name)) node.visible = false;
  });
  scene.add(model);
  const buttons = document.querySelectorAll('[data-view]');
  function frame(view) {
    const face = view === 'face';
    const distance = face ? 1 : Math.max(5.25, 3.7 / camera.aspect);
    controls.target.set(0, face ? 2.33 : 1.28, .015);
    camera.position.set(.12, .045, .99).normalize().multiplyScalar(distance).add(controls.target);
    controls.update();
    buttons.forEach(button => button.setAttribute('aria-pressed', String(button.dataset.view === view)));
    render();
  }
  buttons.forEach(button => { button.onclick = () => frame(button.dataset.view); });
  frame('face');
  status.hidden = true;
}
start().catch(error => {
  console.error(error);
  status.textContent = 'Preview unavailable. Check WebGL 2 and the local model export.';
});
