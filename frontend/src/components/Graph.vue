<template>
  <div id="graph-layer" class="graph-layer">
    <div class="back_blur" hidden> A </div>
    <ul class="graph-grid">
      <li v-for="layer in layerCount" :key="layer">
        <div class="layer-column">
          <NodeSpot
            v-for="nodeIndex in graph.getLayerNodeCount(layer - 1)"
            :key="`${layer - 1}-${nodeIndex - 1}`"
            :layer="layer - 1"
            :index="nodeIndex - 1"
          />
          <!-- Node buat bias -->
          <NodeSpot
            v-if="layer !== layerCount"
            :bias="layer - 1"
          />
          <label>{{ layerLabel(layer - 1) }}</label>
        </div>
      </li>
    </ul>
  </div>
</template>
  
  <script setup>
  import { useGraphStore } from '@/stores/graphStore';
  import Node from './Node.vue';
  import NodeSpot from './NodeSpot.vue';
  import { computed, watch } from 'vue';

  const graph = useGraphStore();

  const layerCount = computed(() => graph.hiddenLayerCount + 1);

  watch(
  () => graph.totalNodeCount,
  () => {
    console.log('Graph now has', graph.totalNodeCount, 'nodes total');
  }
  );

  const layerLabel = (index) => {
    if (index === 0) return "Input";
    if (index === graph.currentGraphState.layers.length - 1) return "Output";
    return `Layer ${index}`;
  };

  </script>
  
  <style scoped>
  .graph-layer {
    position: fixed;
    inset: 0;
    transition: background-color 0.3s;
  }

  ul {
    list-style: none;
    display: flex;

    li {
        margin: 1% 10%;

        div {
            width: 10dvw;
            aspect-ratio: 1/1;
        }
    }
  }

    .graph-grid {
    list-style: none;
    display: flex;
    justify-content: center;
    align-items: flex-start;
    overflow-x: auto;
    padding: 0;
  }

  .layer-column {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin: 0 16px;
  }

  .back_blur {
    width: 100dvw;
    height: 100dvh;
    background-color: rebeccapurple;
    opacity: 25%;
    position: fixed;
    left: 0;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(200px);
  }

  </style>
  