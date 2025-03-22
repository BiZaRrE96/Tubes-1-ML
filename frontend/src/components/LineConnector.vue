<template>
  <svg class="line-layer">
    <!-- Line from 'from' node to 'to' node -->
    <line
      :x1="fromPos.x"
      :y1="fromPos.y"
      :x2="toPos.x"
      :y2="toPos.y"
      stroke="black"
      :stroke-width="strokeWidth"
    />

    <text
      :x="(fromPos.x + toPos.x) / 2 + offsetX"
      :y="(fromPos.y + toPos.y) / 2"
      text-anchor="middle"
      dominant-baseline="middle"
      :fill="textColor"
      font-size="14"
    >
      {{ displayValue }}
    </text>
  </svg>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue';
import { useGraphStore } from '@/stores/graphStore';
import { interpolateColor } from '@/utils/colorUtils.ts';

const props = defineProps<{
  from: { layer: number; index: number };
  to: { layer: number; index: number };
  delta?: number; // Optional delta prop for change in weight
}>();

const store = useGraphStore();

// Calculate positions for from and to nodes
const fromPos = computed(() =>
  store.getNodePos(props.from.layer, props.from.index)
);

const toPos = computed(() =>
  store.getNodePos(props.to.layer, props.to.index)
);

// Get the weight from the store
const weight = computed(() => {
  const fromNode = store.currentGraphState.layers[props.from.layer]?.nodes[props.from.index];
  return fromNode?.weights?.[props.to.index] ?? 0;
});

// Dynamic stroke width based on weight
const strokeWidth = computed(() => Math.max(1, Math.abs(weight.value) * 2));

// Determine which value to display: weight or delta
const displayValue = computed(() => {
  return props.delta !== undefined ? props.delta.toFixed(2) : weight.value.toFixed(2);
});

const good_color = 'green';
const bad_color = 'red';
const neutral_color = 'gray';

// Smooth color transition based on delta (green for good, gray for neutral, red for bad)
const textColor = computed(() => {
  const value = props.delta ?? weight.value; // Use delta if available, else weight
  const goodColor = good_color || 'green';
  const neutralColor = neutral_color || 'gray';
  const badColor = bad_color || 'red';

  if (props.delta !== undefined) {
    // Normalize the delta (assume the delta is in the range -1 to 1)
    const intensity = Math.min(Math.abs(value), 1); // Clamps to 1 (max intensity)

    if (value > 0) {
      // Interpolate between neutral and good color for positive deltas
      return interpolateColor(neutralColor, goodColor, intensity);
    } else if (value < 0) {
      // Interpolate between neutral and bad color for negative deltas
      return interpolateColor(neutralColor, badColor, intensity);
    }
  }

  return neutralColor; // Default neutral color for weight when no delta is specified
});

// Horizontal offset to shift text and avoid overlap
const offsetX = computed(() => {
  const offsetMultiplier = 30;  // Adjust this value to control the spacing between labels
  const totalNodesInLayer = store.getLayerNodeCount(props.from.layer); // Get total nodes in the layer

  if (totalNodesInLayer === 1) return 0; // If only one node, no offset

  // Calculate the ratio based on the index and total nodes in the layer (normalize)
  const ratio = (props.from.index) / (totalNodesInLayer - 1);  // Normalize to range [0, 1]
  
  // Calculate the offset based on the ratio and multiply by the offset multiplier
  const calculatedOffset = ratio * offsetMultiplier - offsetMultiplier / 2; // Center the nodes

  return calculatedOffset;
});


</script>

<style scoped>
.line-layer {
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
  z-index: 1; /* Or wherever it fits below your nodes */
}

line {
  transition: stroke-width 0.2s ease-in-out;
}

text {
  transition: opacity 0.2s ease-in-out;
}
</style>
