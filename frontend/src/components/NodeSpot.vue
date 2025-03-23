<template>
  <div ref="el" class="node-spot">
    <Node
      v-if="savedPos && props.layer !== undefined && props.index !== undefined"
      :layer="props.layer"
      :index="props.index"
      :value="`${props.layer}-${props.index}`"
      :homePosition="targetPos"
    />

    <Node
      v-else-if="savedPos && props.bias !== undefined"
      :bias="props.bias"
      value="Bias"
      :homePosition="targetPos"
    />
  </div>
</template>
  
  <script setup lang="ts">
  import { ref, nextTick, onMounted, onUnmounted , watch, computed} from 'vue';
  import { useGraphStore } from '@/stores/graphStore';
  import Node from './Node.vue';
  
  const props = defineProps<{
    layer?: number;
    index?: number;
    bias?: number; 
  }>();
  
  const el = ref<HTMLDivElement | null>(null);
  const store = useGraphStore();
  const targetPos = ref({ x: 0, y: 0 });
  const savedPos = ref(false);
  
  const updatePosition = () => {
    if (el.value) {
      const rect = el.value.getBoundingClientRect();
      const center = {
        x: rect.left + rect.width / 2,
        y: rect.top + rect.height / 2
      };
      targetPos.value = center;
      
      // Still store this in Pinia too for lines
      if (props.layer !== undefined && props.index !== undefined) {
        store.updatePos(props.layer, props.index, center.x, center.y);
        savedPos.value = true;
        console.log("Layer node ",props.layer,"-",props.index,"updated!!");
      }

      else if (props.bias !== undefined) {
        store.updateBiasPos(props.bias, center.x, center.y);
        savedPos.value = true;
        console.log("Bias node ",props.bias,"updated!!");
      }

      //console.log("Node ",props.layer,"-",props.index,"updated!!");
    }
  };
  
  let resizeObserver: ResizeObserver | null = null;

    onMounted(async () => {
    await nextTick();
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
        updatePosition();
        });
    });

    if (el.value) {
        resizeObserver = new ResizeObserver(() => {
        updatePosition();
        });
        resizeObserver.observe(el.value);
    }
    });

    watch(() => store.totalNodeCount, () => {
    updatePosition();
    });

    watch(() => store.currentGraphState.layers.length, () => {
    updatePosition();
    });

    onUnmounted(() => {
    if (resizeObserver && el.value) {
        resizeObserver.unobserve(el.value);
    }
    });

  </script>

<style scoped>
.node-spot {
    width: 80px;
    aspect-ratio: 1 / 1;
    position: relative;
    border: 1px dashed #ccc;
}
</style>
