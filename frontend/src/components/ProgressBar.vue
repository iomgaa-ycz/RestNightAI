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
import { ref } from 'vue';
const current = ref<number>(1);
const percent = ref<number>(60);
const description = 'This is a description.';
</script>
  
  