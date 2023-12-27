<template>
    <div class="background">
        <a-flex gap="middle" align="start" vertical :style="{ width: '100%' }">
            <div style="height: 20%; width: 100%;"></div>
            <Shower :action="action" />
            <ProgressBar :level="level"/>
            <ConsoleBar @updateLevel="handleUpdateLevel" @updateAction="handleUpdateAction" />
        </a-flex>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import Shower from '@/components/Shower.vue';
import ProgressBar from '@/components/ProgressBar.vue';
import ConsoleBar from '@/components/ConsoleBar.vue';

const level = ref(0);
const action = ref('');

const handleUpdateLevel = (newLevel: number) => {
    level.value = newLevel;
};

const handleUpdateAction = (newAction: string) => {
    action.value = newAction;
};

</script>

<style scoped>
.background {
    /* width: 100%;
    height: 100%; */
    border-radius: 20px; /* 添加border-radius属性 */
    display: flex;
    justify-content: center;
    align-items: center;
}
</style>