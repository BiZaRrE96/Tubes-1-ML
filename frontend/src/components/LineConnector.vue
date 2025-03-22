<template>
  <svg class="line-layer">
    <line
    :x1="fromPos.x"
    :y1="fromPos.y"
    :x2="toPos.x"
    :y2="toPos.y"
    stroke="black"
    :stroke-width="strokeWidth"
  />

  </svg>
</template>

<style>
.line-layer {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 1; /* or wherever it fits below your nodes */
}

</style>

<script setup lang="ts">
import { computed } from 'vue';
import { useGraphStore } from '@/stores/graphStore';

const props = defineProps<{
  from: { layer: number; index: number };
  to: { layer: number; index: number };
}>();

const store = useGraphStore();

const fromPos = computed(() =>
  store.getNodePos(props.from.layer, props.from.index)
);

const toPos = computed(() =>
  store.getNodePos(props.to.layer, props.to.index)
);

const weight = computed(() => {
  const fromNode = store.currentGraphState.layers[props.from.layer]?.nodes[props.from.index];
  return fromNode?.weights?.[props.to.index] ?? 0;
});

const strokeWidth = computed(() => Math.max(1, Math.abs(weight.value) * 2));

</script>
