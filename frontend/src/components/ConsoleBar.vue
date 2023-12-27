<template>
    <a-flex  justify="center" align="center" gap="large" style="width: 100%;">
        <a-button type="primary" :loading="iconLoading" @click="enterIconLoading">
            <template #icon><PoweroffOutlined /></template>
                开始采集
        </a-button>
        <a-button type="primary" @click="enternext">下一步</a-button>
        <a-button type="primary" @click="enterreset">重置</a-button>
    </a-flex>
</template>

<script setup lang="ts">
import { ref, toRefs, defineProps, defineEmits } from 'vue';
import axios from 'axios';

interface DelayLoading {
  delay: number;
}

const iconLoading = ref<boolean | DelayLoading>(false);
const level = ref<number>(0);

// Define the props and emits
const props = defineProps();
const emit = defineEmits(['updateLevel']);

const enterIconLoading = async () => {
  if (iconLoading.value) {
    return;
  }
  try {
    const response = await axios.post('/api/begin_collect', {});

    if (response.status === 200) {
      iconLoading.value = true;
      if (level.value == 0) {
        level.value = 1; // Increment level by 1
        console.log('level updated to: ', level.value);
        emit('updateLevel', level.value); // Emit the updateLevel event
      }
    }
  } catch (error) {
    console.error(error);
  }
};

const enternext = () => {
  iconLoading.value = false;
  if (level.value == 1) {
    level.value = 2; // Increment level by 2
  } else {
    level.value = 0; // Increment level by 0
  }
  emit('updateLevel', level.value); // Emit the updateLevel event
};

const enterreset = () => {
  iconLoading.value = false;
  level.value = 0; // Increment level by 0
  emit('updateLevel', level.value); // Emit the updateLevel event
};

// Export the level variable
const { level: exposedLevel } = toRefs({ level });
</script>