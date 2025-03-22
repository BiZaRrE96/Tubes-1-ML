// stores/graphStore.ts
import { defineStore } from 'pinia';
import { ref, type Ref , computed} from 'vue';
import type { GraphState, Layer, Node } from '../types/graphType';  // Import the shared types

export const useGraphStore = defineStore('graph', () => {
  
    const currentGraphState: Ref<GraphState> = ref({ layers: [] });
    const hiddenLayerCount = ref(1);

  const setHiddenLayerCount = (count: number) => {
    hiddenLayerCount.value = count;
  
    const totalLayerCount = count + 1; // input + hidden + output
    const existing = currentGraphState.value.layers.length;
  
    // Add missing layers
    for (let i = existing; i < totalLayerCount; i++) {
      currentGraphState.value.layers.push({ nodes: [], activation: "linear", bias: 0 });
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
      adjustWeightsForLayer(i);
    }
  };
  
  

  const addNode = (layer: number) => {
    const nextLayer = currentGraphState.value.layers[layer + 1];
  
    // Create a new node with weights corresponding to the number of nodes in the next layer
    const newNode: Node = {
      weights: nextLayer ? Array(nextLayer.nodes.length).fill(1.0) : [], // Initialize weights for new node
      x: 0,
      y: 0
    };
  
    currentGraphState.value.layers[layer].nodes.push(newNode);
  
    // Now adjust weights for the other nodes in the layer based on the number of nodes in the next layer
    adjustWeightsForLayer(layer-1);
  };
  
  const popNode = (layer: number) => {
    if (currentGraphState.value.layers[layer]?.nodes.length > 0) {
      currentGraphState.value.layers[layer].nodes.pop();
      // Adjust weights for the remaining nodes
      adjustWeightsForLayer(layer-1);
    }
  };
  
  // Function to adjust weights based on the number of nodes in the next layer
  const adjustWeightsForLayer = (layer: number) => {
    if (layer < 0) {
      return;
    }
    const nextLayer = currentGraphState.value.layers[layer + 1];
  
    // Iterate through each node and add weights if necessary
    currentGraphState.value.layers[layer].nodes.forEach((node) => {
      const currentWeightCount = node.weights?.length || 0;
      const nextLayerNodeCount = nextLayer?.nodes.length || 0;
  
      // If the weight count is less than the next layer node count, append more weights
      if (currentWeightCount < nextLayerNodeCount) {
        const additionalWeights = Array(nextLayerNodeCount - currentWeightCount).fill(1.0); // Default to 1.0
        node.weights = [...(node.weights || []), ...additionalWeights];
      }
    });
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

  const updateLayerActivation = (layerIndex: number, activation: string) => {
    if (currentGraphState.value.layers[layerIndex]) {
      currentGraphState.value.layers[layerIndex].activation = activation;
    }
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
  console.log("Setup clear",currentGraphState.value)

  return {
    currentGraphState,
    hiddenLayerCount,
    setHiddenLayerCount,
    updatePos,
    updateLayerActivation,
    getNodePos,
    getLayerInfo,
    getLayerNodeCount,
    addNode,
    popNode,
    totalNodeCount,
  };  
});
