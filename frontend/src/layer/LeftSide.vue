<template>
    <div class="background">
        <h1 class="word">睡姿体动检测</h1>
    </div>
</template>

<script setup lang="ts">
import { ref } from 'vue';

</script>

<style scoped>
.background {
    background-color: #f5f5f5;
    width: 100%;
    height: 15vh;
    border-radius: 20px; /* 添加border-radius属性 */
    display: flex;
    justify-content: center;
    align-items: center;
}
.word {
    font-size: 50px;
    font-weight: bold;
    color: #000000;
}
</style>