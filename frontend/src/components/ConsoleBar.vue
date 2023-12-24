<template>
    <a-flex  justify="center" align="center" gap="large" style="width: 100%;">
        <a-button type="primary" :loading="iconLoading" @click="enterIconLoading">
            <template #icon><PoweroffOutlined /></template>
                开始采集
        </a-button>
        <a-button type="primary">下一步</a-button>
        <a-button type="primary">重置</a-button>
    </a-flex>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import axios from 'axios';
interface DelayLoading {
  delay: number;
}
const iconLoading = ref<boolean | DelayLoading>(false);

const enterIconLoading = async () => {
  iconLoading.value = true;
  try {
    const response = await axios.post('/api/begin_collect', {});

    if (response.status === 200) {
      iconLoading.value = false;
    }
  } catch (error) {
    console.error(error);
    iconLoading.value = false;
  }
};
</script>


<style scoped>
.background {
    border-radius: 20px; /* 添加border-radius属性 */
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>