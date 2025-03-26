// graphInterface.ts

import type { GraphState } from '../types/graphType';  // Import GraphState from the shared types
import { useGraphStore } from '@/stores/graphStore';

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
    const store = useGraphStore();
    store.currentGraphState = data;
    store.hiddenLayerCount = data.layers.length - 1;
    return data;
  } catch (error) {
    console.error('Error fetching graph from backend:', error);
    return null
  }
};


// Export graph data to file (download the graph file from backend)
export const exportGraph = async (): Promise<void> => {
  try {
    const response = await fetch('http://localhost:5000/api/export', {
      method: 'POST',
    });
    
    if (response.ok) {
      const blob = await response.blob();  // Get the file blob
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'graph_model.pkl';  // Name of the file to download
      a.click();
      window.URL.revokeObjectURL(url);
    } else {
      const data = await response.json();
      console.error('Error exporting graph:', data.error);
    }
  } catch (error) {
    console.error('Error exporting graph:', error);
  }
};

// Import graph data from file (upload a graph file to the backend)
const importGraph = async (file: File): Promise<GraphState | null> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:5000/api/import', {
      method: 'POST',
      body: formData,  // Send the file as form data
    });

    const data: GraphState = await response.json();  // Use GraphState to type the response
    if (response.ok) {
      return data;
    } else {
      console.error('Error importing graph:', response.statusText);
      return null;
    }
  } catch (error) {
    console.error('Error importing graph:', error);
    return null;
  }
};

export const importFromFile = async () => {
  // Create a file input element dynamically
  const input = document.createElement('input');
  input.type = 'file';  // Set the input type to 'file'

  // Trigger the file dialog when the input is clicked
  input.click();

  // When a file is selected, import it
  input.onchange = async () => {
    const file = input.files?.[0];  // Get the selected file

    if (file) {
      // Call the importGraph function and pass the selected file
      const data = await importGraph(file);
      console.log('Imported graph:', data);
    }
  };
};

export const initializeWeightsOnBackend = async (): Promise<void> => {
  try {
    const response = await fetch('http://localhost:5000/api/initialize_weights', {
      method: 'POST',
    });
    const data = await response.json();
    console.log('Weights initialized:', data);
  } catch (error) {
    console.error('Error initializing weights on backend:', error);
  }
};

export const startLearningOnBackend = async (learningRate: number): Promise<void> => {
  try {
    const response = await fetch('http://localhost:5000/api/start_learning', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ learningRate }),
    });
    const data = await response.json();
    console.log('Learning started:', data);
  } catch (error) {
    console.error('Error starting learning on backend:', error);
  }
};