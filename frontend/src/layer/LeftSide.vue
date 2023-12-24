<template>
    <div class="background">
        <CollectList :data="Record" />
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import CollectList from '@/components/CollectList.vue';

const Record = [
  { id: '1', labels: '标签1' },
  { id: '2', labels: '标签2' },
];

</script>

<style scoped>
.background {
    width: 100%;
    height: 100%;
    border-radius: 20px; /* 添加border-radius属性 */
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>