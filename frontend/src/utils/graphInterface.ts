// graphInterface.ts

import type { GraphState } from '../types/graphType';  // Import GraphState from the shared types

// Function to send config to the backend
export const sendConfigToBackend = async (graph: GraphState): Promise<any> => {
  try {
    const response = await fetch('http://localhost:5000/api/train', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(graph),  // Send the graph as a GraphState object
    });
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error sending config to backend:', error);
  }
};

// Function to get graph from the backend (simulated for now)
export const getGraphFromBackend = async (): Promise<GraphState | null> => {
  try {
    const response = await fetch('http://localhost:5000/api/get_graph', {
      method: 'GET',
    });
    const data: GraphState = await response.json();  // Use GraphState to type the response
    return data;
  } catch (error) {
    console.error('Error fetching graph from backend:', error);
    return null
  }
};

// Export graph data to file
export const exportGraph = async (): Promise<void> => {
  try {
    const response = await fetch('http://localhost:5000/api/export', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('Exported graph:', data);
  } catch (error) {
    console.error('Error exporting graph:', error);
  }
};

// Import graph data from file
export const importGraph = async (): Promise<GraphState | null> => {
  try {
    const response = await fetch('http://localhost:5000/api/import', {
      method: 'POST',
    });
    const data: GraphState = await response.json();  // Use GraphState to type the response
    return data;
  } catch (error) {
    console.error('Error importing graph:', error);
    return null
  }
};
