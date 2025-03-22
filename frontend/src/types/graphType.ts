// graphTypes.ts (Shared Types for Frontend and Backend)

export interface Node {
    weights?: number[];  // Weights of the node (connections to the next layer)
    x?: number;          // X coordinate (for visualization purposes)
    y?: number;          // Y coordinate (for visualization purposes)
  }
  
  export interface Layer {
    nodes: Node[];       // Nodes in the layer
    bias: number;        // Bias for the layer (e.g., for each node or as a whole)
    activation?: string; // Activation function for the layer
  }
  
  export interface GraphState {
    layers: Layer[];     // Layers in the graph
  }
  