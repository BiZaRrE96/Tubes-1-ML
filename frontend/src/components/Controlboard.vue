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

    <div v-for="(layer, index) in visibleLayers" :key="index">
      <details>
        <summary>
          <strong>
            {{ index === 0 ? 'Input Layer' : `Hidden Layer ${index}` }}
          </strong>
          <button @click="graph.addNode(index)">+</button>
          <button @click="graph.popNode(index)">-</button>
          <span>{{ graph.getLayerNodeCount(index) }} nodes</span>
        </summary>

        <label>
          Activation Function:
          <select v-model="activationModels[index]" @change="updateActivation(index)">
            <option value="linear">Linear</option>
            <option value="relu">ReLU</option>
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="softmax">Softmax</option>
          </select>
        </label>

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

      <details v-if="index < visibleLayers.length - 1">
        <summary>
          <strong>Bias Layer {{ index }} - {{ index + 1 }}  </strong>
          
        </summary>

        <div
          v-for="(weight, weightIndex) in graph.currentGraphState.biases[index].weights"
          :key="`bias-${index}-${weightIndex}`"
        >
          → to node {{ weightIndex }} in layer {{ index + 1 }}:
          <input
            type="number"
            step="0.1"
            v-model.number="graph.currentGraphState.biases[index].weights[weightIndex]"
          />
        </div>
      </details>
    </div>

    <!-- Buttons to trigger actions -->
    <button @click="sendToBackend">Send to Backend</button>
    <button @click="getFromBackend">Get from Backend</button>
    <button @click="importFromFile">Import from File</button>
    <button @click="exportToFile">Export to File</button>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';
import { useGraphStore } from '@/stores/graphStore';
import { sendConfigToBackend, getGraphFromBackend, exportGraph, importFromFile as iff } from '@/utils/graphInterface'; // Assuming utils

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
  graph.currentGraphState.layers //.slice(0, graph.currentGraphState.layers.length - 1)
);

// A reactive object to store selected activation functions for each layer
const activationModels = ref<{ [key: number]: string }>({});

// Function to get the activation for the layer (with a fallback of 'linear')
const getActivation = (index: number): string => {
  return graph.currentGraphState.layers[index].activation ?? 'linear'; // Default to 'linear'
};

// Function to update the activation of a layer
const updateActivation = (index: number) => {
  const selectedActivation = activationModels.value[index];
  graph.updateLayerActivation(index, selectedActivation);
};

// Actions for buttons

const sendToBackend = async () => {
  console.log(graph.currentGraphState);
  const config = sendConfigToBackend(graph.currentGraphState);
  console.log('Sent graph to backend:', config);
};

const getFromBackend = async () => {
  const data = await getGraphFromBackend();
  
  console.log('Received graph from backend:', data);
};

const importFromFile = async () => {
  const data = await iff();
  console.log('Imported graph:', data);
};

const exportToFile = async () => {
  await exportGraph();
  console.log('Exported graph');
};

graph.currentGraphState.layers.forEach((layer, index) => {
  activationModels.value[index] = layer.activation ?? 'linear'; // Default to 'linear'
});
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

button {
  margin-top: 10px;
  padding: 5px;
  cursor: pointer;
}
</style>
