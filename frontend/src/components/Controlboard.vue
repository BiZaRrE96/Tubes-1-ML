<template>
  <div
    class="control-board"
    :style="{ top: position.y + 'px', left: position.x + 'px' }"
    @mousedown="startDrag"
  >
    <label>
      Hidden Layers:
      <input
        type="range"
        min="1"
        max="10"
        step="1"
        :value="graph.hiddenLayerCount"
        @input="graph.setHiddenLayerCount(Number($event.target.value))"
      />
      <output>{{ graph.hiddenLayerCount }}</output>
    </label>

    <hr />
    <div
      v-for="(layer, index) in visibleLayers"
      :key="index"
    >
      <details>
        <summary>
          <strong>
            {{ index === 0 ? 'Input Layer' : `Hidden Layer ${index}` }}
          </strong>
          <button @click.stop="graph.addNode(index)">+</button>
          <button @click.stop="graph.popNode(index)">-</button>
          <span>{{ graph.getLayerNodeCount(index) }} nodes</span>
        </summary>

        <div
          v-for="(node, nodeIndex) in graph.getLayerInfo(index).nodes"
          :key="`node-${index}-${nodeIndex}`"
          class="node-weight-editor"
        >
          <h4>Node {{ nodeIndex }}</h4>
          <div
            v-for="(weight, weightIndex) in node.weights ?? []"
            :key="`weight-${index}-${nodeIndex}-${weightIndex}`"
          >
            → to node {{ weightIndex }} in layer {{ index + 1 }}:
            <input
              type="number"
              step="0.1"
              v-model.number="graph.currentGraphState.layers[index].nodes[nodeIndex].weights[weightIndex]"
            />
          </div>
        </div>
      </details>
    </div>

  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useGraphStore } from '@/stores/graphStore';

const graph = useGraphStore();

const position = ref({ x: 50, y: 50 });
let dragging = false;
let offset = { x: 0, y: 0 };

const startDrag = (e: MouseEvent) => {
  if (['INPUT', 'TEXTAREA', 'BUTTON', 'SELECT', 'LABEL'].includes(e.target.tagName)) return;

  dragging = true;
  offset.x = e.clientX - position.value.x;
  offset.y = e.clientY - position.value.y;

  window.addEventListener('mousemove', onDrag);
  window.addEventListener('mouseup', stopDrag);
};

const onDrag = (e: MouseEvent) => {
  if (!dragging) return;
  position.value.x = e.clientX - offset.x;
  position.value.y = e.clientY - offset.y;
};

const stopDrag = () => {
  dragging = false;
  window.removeEventListener('mousemove', onDrag);
  window.removeEventListener('mouseup', stopDrag);
};

const visibleLayers = computed(() =>
  graph.currentGraphState.layers.slice(0, graph.currentGraphState.layers.length - 1)
);


</script>

<style scoped>
.control-board {
  position: absolute;
  background-color: var(--color-background-mute);
  color: var(--color-text);
  border: 1px solid #ccc;
  padding: 1rem;
  cursor: grab;
  user-select: none;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  max-width: 300px;
  z-index: 10;
}

label {
  display: block;
  margin-bottom: 1rem;
}

input[type='range'] {
  margin-left: 0.5rem;
}

output {
  font-weight: bold;
  margin-left: 0.5rem;
}

.node-weight-editor {
  margin-left: 1rem;
  padding: 0.25rem 0.5rem;
  background-color: var(--color-background-soft);
  border: 1px solid var(--color-border);
  border-radius: 4px;
  margin-bottom: 0.5rem;
}
.node-weight-editor input[type='number'] {
  width: 60px;
  margin-left: 0.5rem;
}

</style>
