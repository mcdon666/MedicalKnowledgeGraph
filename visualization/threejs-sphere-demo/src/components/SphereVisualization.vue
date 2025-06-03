<template>
  <div ref="container" class="container"></div>
</template>

<script setup>
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls'
import { onMounted, onBeforeUnmount, ref } from 'vue'

// 容器引用
const container = ref(null)

// 黄金分割球面点生成均匀点
function fibonacciSphere(samples) {
  let points = [];
  const phi = Math.PI * (3 - Math.sqrt(5)); // 黄金角
  for (let i = 0; i < samples; i++) {
    const y = 1 - (i / (samples - 1)) * 2;
    const radius = Math.sqrt(1 - y * y);
    const theta = phi * i;
    const x = Math.cos(theta) * radius;
    const z = Math.sin(theta) * radius;
    points.push(new THREE.Vector3(x, y, z));
  }
  return points;
}

const spheres = []

let scene, camera, renderer, controls, raycaster, mouse

let hoveredSphere = null
let label = null

async function fetchDepartments() {
  const res = await fetch('http://localhost:4000/api/departments')
  if (!res.ok) throw new Error('Failed to fetch')
  return await res.json()
}

onMounted(async () => {
  // 初始化场景
  scene = new THREE.Scene()

  let departments = []
  try {
    departments = await fetchDepartments()
  } catch (e) {
    console.error(e)
    departments = ["无法加载部门"]  // 失败时展示
  }

  // 相机
  const width = container.value.clientWidth
  const height = container.value.clientHeight
  camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000)
  camera.position.z = 4

  // 渲染器
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(width, height)
  container.value.appendChild(renderer.domElement)

  // 控制器
  controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true
  controls.dampingFactor = 0.05
  controls.enablePan = false
  controls.minDistance = 2
  controls.maxDistance = 10

  // 光源
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7)
  scene.add(ambientLight)
  const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
  directionalLight.position.set(5,5,5)
  scene.add(directionalLight)

  // 添加节点球体
  const points = fibonacciSphere(departments.length)
  const geometry = new THREE.SphereGeometry(0.05, 16, 16)
  const material = new THREE.MeshStandardMaterial({ color: 0x00aaff })

  points.forEach((pos, i) => {
    const sphere = new THREE.Mesh(geometry, material.clone())
    sphere.position.copy(pos.multiplyScalar(2))  // 放大球体半径为2
    sphere.userData = { name: departments[i] }
    scene.add(sphere)
    spheres.push(sphere)
  })

  // 射线投射用于交互
  raycaster = new THREE.Raycaster()
  mouse = new THREE.Vector2()

  // 鼠标点击事件
  renderer.domElement.addEventListener('click', onClick)

  // 鼠标移动事件
  createLabel()

  renderer.domElement.addEventListener('mousemove', onMouseMove)
  renderer.domElement.addEventListener('mouseleave', onMouseLeave)

  // 响应窗口resize
  window.addEventListener('resize', onResize)

  animate()
})

function animate() {
  requestAnimationFrame(animate)
  controls.update()
  renderer.render(scene, camera)
}

function onResize() {
  const width = container.value.clientWidth
  const height = container.value.clientHeight
  camera.aspect = width / height
  camera.updateProjectionMatrix()
  renderer.setSize(width, height)
}

function onClick(event) {
  // 计算鼠标位置标准化设备坐标
  const rect = renderer.domElement.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  raycaster.setFromCamera(mouse, camera)
  const intersects = raycaster.intersectObjects(spheres)
  if (intersects.length > 0) {
    const clicked = intersects[0].object
    alert(`你点击了部门: ${clicked.userData.name}`)
  }
}


// 初始化label，只创建一次
function createLabel() {
  label = document.createElement('div')
  label.style.position = 'absolute'
  label.style.padding = '4px 8px'
  label.style.backgroundColor = 'rgba(0,0,0,1)'
  label.style.color = 'white'
  label.style.fontWeight = 'bold'
  label.style.borderRadius = '4px'
  label.style.pointerEvents = 'none'
  label.style.fontSize = '16px'
  label.style.whiteSpace = 'nowrap'
  label.style.display = 'none'
  document.body.appendChild(label)
  console.log('label created and appended')
}


function onMouseMove(event) {
  const rect = renderer.domElement.getBoundingClientRect()
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1

  raycaster.setFromCamera(mouse, camera)
  const intersects = raycaster.intersectObjects(spheres)

  if (intersects.length > 0) {
    const sphere = intersects[0].object

    // 放大球体
    spheres.forEach(s => s.scale.set(1, 1, 1)) // 重置所有
    sphere.scale.set(1.5, 1.5, 1.5)

    // 显示标签
    if (label) {
      label.style.display = 'block'
      label.textContent = sphere.userData.name

      // 计算标签位置（球体 -> 屏幕坐标）
      const vector = sphere.position.clone().project(camera)
      const screenX = (vector.x * 0.5 + 0.5) * rect.width + rect.left
      const screenY = (-vector.y * 0.5 + 0.5) * rect.height + rect.top
      label.style.left = `${screenX}px`
      label.style.top = `${screenY}px`
    }

  } else {
    // 恢复大小、隐藏标签
    spheres.forEach(s => s.scale.set(1, 1, 1))
    if (label) label.style.display = 'none'
  }
}



function onMouseLeave() {
  // 鼠标离开画布时恢复状态
  if (hoveredSphere) hoveredSphere.scale.set(1,1,1)
  hoveredSphere = null
  if (label) label.style.display = 'none'
}

onBeforeUnmount(() => {
  renderer.domElement.removeEventListener('click', onClick)
  window.removeEventListener('resize', onResize)
  renderer.dispose()
  
  renderer.domElement.removeEventListener('mousemove', onMouseMove)
})
</script>

<style scoped>
.container {
  position: relative;
  width: 100%;
  height: 100vh;
  background-color: #111;
  overflow: hidden;
}

</style>
