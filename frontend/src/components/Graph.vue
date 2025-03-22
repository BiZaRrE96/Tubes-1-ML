<template>
    <div id="graph-layer" class="graph-layer">
        <ul>
            <li v-for="layer in layerCount" :key="layer">
                <NodeSpot
                    v-for="nodeIndex in graph.getLayerNodeCount(layer - 1)"
                    :key="`${layer - 1}-${nodeIndex - 1}`"
                    :layer="layer - 1"
                    :index="nodeIndex - 1"
                    />
                <label>{{ layer - 1 === 0 ? "Input" : layer == layerCount ? "Output" : `Layer ${layer - 1}` }}</label>
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
  </script>
  
  <style scoped>
  .graph {
    position: fixed;
    inset: 0;
    transition: background-color 0.3s;
  }

  ul {
    list-style: none;
    display: flex;

    li {
        margin: 10%;

        div {
            width: 10dvw;
            aspect-ratio: 1/1;
        }
    }
  }
  </style>
  