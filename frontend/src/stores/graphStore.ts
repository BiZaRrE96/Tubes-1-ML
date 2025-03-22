// stores/graphStore.ts
import { defineStore } from 'pinia';
import { ref, type Ref , computed} from 'vue';

export const useGraphStore = defineStore('graph', () => {
  
    const currentGraphState: Ref<GraphState> = ref({ layers: [] });
    const hiddenLayerCount = ref(1);

  const setHiddenLayerCount = (count: number) => {
    hiddenLayerCount.value = count;
  
    const totalLayerCount = count + 2; // input + hidden + output
    const existing = currentGraphState.value.layers.length;
  
    // Add missing layers
    for (let i = existing; i < totalLayerCount; i++) {
      currentGraphState.value.layers.push({ nodes: [], bias: 0 });
    }
  
    // Remove extra layers
    while (currentGraphState.value.layers.length > totalLayerCount) {
      currentGraphState.value.layers.pop();
    }
  
    // Auto-adjust weights for all nodes
    updateAllWeights();
  };
  
  const updateAllWeights = () => {
    const layers = currentGraphState.value.layers;
    for (let i = 0; i < layers.length - 1; i++) {
      const current = layers[i];
      const next = layers[i + 1];
      current.nodes.forEach((node) => {
        node.weights = Array(next.nodes.length).fill(1.0); // default to 1.0
      });
    }
  };
  
  

  const addNode = (layer: number) => {
    const nextLayer = currentGraphState.value.layers[layer + 1];
    const newNode: Node = {
      weights: nextLayer ? Array(nextLayer.nodes.length).fill(1.0) : [],
      x: 0,
      y: 0
    };
  
    currentGraphState.value.layers[layer].nodes.push(newNode);
    updateAllWeights(); // Refresh weights
  };
  
  const popNode = (layer: number) => {
    if (currentGraphState.value.layers[layer]?.nodes.length > 0) {
      currentGraphState.value.layers[layer].nodes.pop();
      updateAllWeights();
    }
  };
  

  // ✅ Update node position
  const updatePos = (layer: number, index: number, x: number, y: number) => {
    //console.log('updatePos', layer, index, x, y);
    const node = currentGraphState.value.layers[layer]?.nodes[index];
    if (node) {
      node.x = x;
      node.y = y;
    }
  };

  // ✅ Get node position
  const getNodePos = (layer: number, index: number): { x: number; y: number } => {
    const node = currentGraphState.value.layers[layer]?.nodes[index];
    //console.log('getNodePos', layer, index, "| ", node.x, node.y);
    return {
      x: Number(node?.x ?? 0),
      y: Number(node?.y ?? 0)
    };
  };

  const getLayerInfo = (layer: number): Layer => {
    return currentGraphState.value.layers[layer];
};

    const getLayerNodeCount = (layer: number): number => {
        if (layer > currentGraphState.value.layers.length) {
            return 0;
        }
        return currentGraphState.value.layers[layer].nodes.length;
  };

  const totalNodeCount = computed(() => {
    return currentGraphState.value.layers.reduce((total, layer) => {
      return total + layer.nodes.length;
    }, 0);
  });
  


  // Later: undo/redo support here
  let history: GraphState[];

  setHiddenLayerCount(hiddenLayerCount.value);

  return {
    currentGraphState,
    hiddenLayerCount,
    setHiddenLayerCount,
    updatePos,
    getNodePos,
    getLayerInfo,
    getLayerNodeCount,
    addNode,
    popNode,
    totalNodeCount,
  };  
});

// Type definitions
interface GraphState {
  layers: Layer[];
}

interface Layer {
  nodes: Node[];
  bias: number;
}

interface Node {
  weights?: number[];
  x?: number;
  y?: number;
}
