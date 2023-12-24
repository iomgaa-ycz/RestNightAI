<template>
    <a-steps
      v-model:current="current"
      :percent="percent"
      :items="[
        {
          title: '准备',
          description,
        },
        {
          title: '采集',
          subTitle: 'Left 00:00:08',
          description,
        },
        {
          title: '结束',
          description,
        },
      ]"
    ></a-steps>
</template>
<script lang="ts" setup>
import { ref, onMounted, watch } from 'vue';
import { watchEffect } from 'vue';
const current = ref<number>(1);
const percent = ref<number>(60);
const description = '';

const props = defineProps({
  level: {
    type: Number,
    required: true,
  },
});

// Update current and percent based on level prop
watchEffect(() => {
  console.log(props.level);
  if (props.level === 0) {
    current.value = 0;
  } else if (props.level === 1) {
    current.value = 1;
  } else if (props.level === 2) {
    current.value = 2;
  }
});

</script>
  
  