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
import send from '../TS/http';
interface DelayLoading {
  delay: number;
}
const iconLoading = ref<boolean | DelayLoading>(false);

const enterIconLoading = () => {
  send("/api/begin_collect",{});

  setTimeout(() => {
    iconLoading.value = false;
  }, 6000);
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