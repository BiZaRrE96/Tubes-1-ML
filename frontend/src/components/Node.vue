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
  <LineConnector v-if="props.layer !== undefined && props.index !== undefined && props.layer < store.hiddenLayerCount"
    v-for="target in nextTargets"
    :key="`from-${props.layer}-${props.index}-to-${target.index}`"
    :from="{ layer: props.layer, index: props.index }"
    :to="target"
  />

  <LineConnector v-else-if="props.bias !== undefined"
    v-for="target in nextTargets"
    :key="`from-bias-${props.bias}-to-${target.index}`"
    :from="{ bias: props.bias }"
    :to="target"
  />
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watchEffect, nextTick } from 'vue';
import { useGraphStore } from '@/stores/graphStore';
import LineConnector from './LineConnector.vue';

const props = defineProps<{
  layer?: number;
  index?: number;
  bias?: number;
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

  const nextLayer = props.layer?.valueOf() !== undefined ? props.layer + 1 : props.bias?.valueOf() !== undefined ? props.bias + 1 : -1;

  console.log("Node setup of",props.value, "Next layers :", nextLayer, store.currentGraphState.layers.length);

  if (nextLayer >= store.currentGraphState.layers.length || nextLayer < 0) {
    nextTargets.value = []; // no lines from output layer
    return;
  }

  const count = store.getLayerNodeCount(nextLayer);
  nextTargets.value = [];

  for (let i = 0; i < count; i++) {
    nextTargets.value.push({ layer: nextLayer, index: i });
  }

  console.log("Nodesetup of",props.value,nextLayer, count, nextTargets.value);
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

  if (props.layer !== undefined && props.index !== undefined) {
    store.updatePos(props.layer, props.index, position.value.x, position.value.y);
  } else if (props.bias !== undefined) {
    store.updateBiasPos(props.bias, position.value.x, position.value.y);
  }
};

// 🟢 Stop drag
const stopDrag = () => {
  dragging.value = false;

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

      if (props.layer !== undefined && props.index !== undefined) {
        store.updatePos(props.layer, props.index, position.value.x, position.value.y);
      } else if (props.bias !== undefined) {
        store.updateBiasPos(props.bias, position.value.x, position.value.y);
      }
    }
  }

  rafId = requestAnimationFrame(driftBack);
};

// 🔃 On mount
onMounted(async () => {
  await nextTick();
  console.log("Node setup of",props);
  position.value = { ...props.homePosition };
  rafId = requestAnimationFrame(driftBack);
  if (props.layer !== undefined && props.index !== undefined) {
    store.updatePos(props.layer, props.index, position.value.x, position.value.y);
  } else if (props.bias !== undefined) {
    store.updateBiasPos(props.bias, position.value.x, position.value.y);
  }
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
