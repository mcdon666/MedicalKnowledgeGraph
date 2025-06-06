import * as THREE from "https://esm.sh/three";
import { GLTFLoader } from "https://esm.sh/three/examples/jsm/loaders/GLTFLoader.js";
import { OrbitControls } from "https://esm.sh/three/examples/jsm/controls/OrbitControls.js";
import { RGBELoader } from "https://esm.sh/three/examples/jsm/loaders/RGBELoader.js";

let partCategories = null;

async function loadPartCategories() {
  const response = await fetch("partCategories.json");
  if (!response.ok) {
    throw new Error("Failed to load partCategories.json");
  }
  partCategories = await response.json();
}

async function init() {
  await loadPartCategories();

  const canvas = document.getElementById("canvas");
  const container = canvas.parentElement;

  // 初始化渲染器
  const renderer = new THREE.WebGLRenderer({
    canvas: canvas,
    antialias: true,
    alpha: true,
  });

  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.1;
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  // 场景 & 相机
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    75,
    container.clientWidth / container.clientHeight,
    0.1,
    1000
  );
  camera.position.set(0, 0, 1.5);

  // 控制器
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;

  // 灯光
  const light = new THREE.DirectionalLight(0xffffff, 1);
  light.castShadow = true;
  light.shadow.mapSize.width = 2048;
  light.shadow.mapSize.height = 2048;
  light.shadow.radius = 4;
  scene.add(light);

  const ambientLight = new THREE.AmbientLight(0xffffff, 2);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 4);
  directionalLight.position.set(5, 10, 7);
  scene.add(directionalLight);

  // 环境贴图
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  new RGBELoader()
    .setDataType(THREE.UnsignedByteType)
    .load(
      "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/studio_small_08_2k.hdr",
      (texture) => {
        const envMap = pmremGenerator.fromEquirectangular(texture).texture;
        scene.environment = envMap;
        // scene.background = envMap;
        texture.dispose();
        pmremGenerator.dispose();
      }
    );

  // 高亮材质
  const highlightMaterial = new THREE.MeshStandardMaterial({
    color: 0x00ff00,
    emissive: 0x00ff00,
    emissiveIntensity: 0.5,
    metalness: 0.8,
    roughness: 0.7,
  });

  let currentHighlightGroup = [];
  let originalMaterials = new Map();

  // 加载模型
  const loader = new GLTFLoader();
  loader.load(
    "Model.glb",
    (gltf) => {
      const model = gltf.scene;
      scene.add(model);

      const box = new THREE.Box3().setFromObject(model);
      const center = box.getCenter(new THREE.Vector3());
      model.position.sub(center);

      model.traverse((child) => {
        if (child.isMesh) {
          child.userData.originalMaterial = child.material.clone();
          for (const [category, parts] of Object.entries(partCategories)) {
            if (parts.includes(child.name)) {
              child.userData.partCategory = category;
              break;
            }
          }
          if (!child.userData.partCategory) {
            child.userData.partCategory = "面部";
          }

          if (child.material.map) {
            child.material.map.anisotropy =
              renderer.capabilities.getMaxAnisotropy();
          }
        }
      });
    },
    undefined,
    (error) => {
      console.error("模型加载失败:", error);
    }
  );

  // 射线检测
  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  function restoreHighlight() {
    currentHighlightGroup.forEach((obj) => {
      obj.material = obj.userData.originalMaterial;
    });
    currentHighlightGroup = [];
  }

  function highlightGroup(object) {
    restoreHighlight();
    const category = object.userData.partCategory;
    scene.traverse((child) => {
      if (child.isMesh && child.userData.partCategory === category) {
        originalMaterials.set(child, child.material);
        child.material = highlightMaterial;
        currentHighlightGroup.push(child);
      }
    });
  }

  window.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);
    if (intersects.length > 0 && intersects[0].object.isMesh) {
      highlightGroup(intersects[0].object);
    } else {
      restoreHighlight();
    }
  });

  window.addEventListener("click", (event) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(mouse, camera);
    const intersects = raycaster.intersectObjects(scene.children, true);
    if (intersects.length > 0 && intersects[0].object.isMesh) {
      const displayElement = document.getElementById("part-name-display");
      if (displayElement) {
        displayElement.textContent = intersects[0].object.userData.partCategory;
        // console.log("点击了部位: " + intersects[0].object.userData.partCategory);
      }
      // console.log(intersects[0].object.userData.partCategory)
      // alert("点击了部位: " + intersects[0].object.userData.partCategory);
    }
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  // Resize
  window.addEventListener("resize", () => {
    const width = container.clientWidth;
    const height = container.clientHeight;
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  });
}

init();
