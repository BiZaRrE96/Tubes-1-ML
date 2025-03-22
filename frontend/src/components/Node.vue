<template>
  <div
    class="draggable-node"
    :style="{
      top: position.y - size / 2 + 'px',
      left: position.x - size / 2 + 'px',
      width: size + 'px',
      height: size + 'px'
    }"
    @mousedown="startDrag"
  >
    {{ displayValue }}
  </div>

  <LineConnector
    v-for="target in nextTargets"
    :key="`from-${props.layer}-${props.index}-to-${target.index}`"
    :from="{ layer: props.layer, index: props.index }"
    :to="target"
  />
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watchEffect } from 'vue';
import { useGraphStore } from '@/stores/graphStore';
import LineConnector from './LineConnector.vue';

const props = defineProps<{
  layer: number;
  index: number;
  value?: number | string;
  homePosition: {
    x: number;
    y: number;
  };
}>();

const store = useGraphStore();
const displayValue = props.value ?? ' ';
const size = 40;

const position = ref({ x: 0, y: 0 });
const dragging = ref(false);
let offset = { x: 0, y: 0 };

// 🔁 Reactive list of next-layer targets
const nextTargets = ref<{ layer: number; index: number }[]>([]);

watchEffect(() => {
  const nextLayer = props.layer + 1;

  if (nextLayer >= store.currentGraphState.layers.length) {
    nextTargets.value = []; // no lines from output layer
    return;
  }

  const count = store.getLayerNodeCount(nextLayer);
  nextTargets.value = [];

  for (let i = 0; i < count; i++) {
    nextTargets.value.push({ layer: nextLayer, index: i });
  }
});

// 🟢 Start drag
const startDrag = (e: MouseEvent) => {
  dragging.value = true;
  offset.x = e.clientX - position.value.x;
  offset.y = e.clientY - position.value.y;

  document.body.style.userSelect = 'none';
  window.addEventListener('mousemove', onDrag);
  window.addEventListener('mouseup', stopDrag);
};

// 🟢 On drag
const onDrag = (e: MouseEvent) => {
  if (!dragging.value) return;
  position.value.x = e.clientX - offset.x;
  position.value.y = e.clientY - offset.y;

  store.updatePos(props.layer, props.index, position.value.x, position.value.y);
};

// 🟢 Stop drag
const stopDrag = () => {
  dragging.value = false;
  store.updatePos(props.layer, props.index, position.value.x, position.value.y);

  document.body.style.userSelect = '';
  window.removeEventListener('mousemove', onDrag);
  window.removeEventListener('mouseup', stopDrag);
};

// 🔁 Drift logic
let rafId: number;
const driftBack = () => {
  if (!dragging.value) {
    const dx = props.homePosition.x - position.value.x;
    const dy = props.homePosition.y - position.value.y;

    if (Math.abs(dx) > 0.5 || Math.abs(dy) > 0.5) {
      position.value.x += dx * 0.1;
      position.value.y += dy * 0.1;

      store.updatePos(props.layer, props.index, position.value.x, position.value.y);
    }
  }

  rafId = requestAnimationFrame(driftBack);
};

// 🔃 On mount
onMounted(() => {
  position.value = { ...props.homePosition };
  rafId = requestAnimationFrame(driftBack);
});

onUnmounted(() => cancelAnimationFrame(rafId));
</script>

<style scoped>
.draggable-node {
  position: fixed;
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background-color: lightblue;
  color: black;
  font-size: 14px;
  text-align: center;
  line-height: 40px;
  border: 1px solid black;
  cursor: grab;
  user-select: none;
  z-index: 5;
}
</style>
