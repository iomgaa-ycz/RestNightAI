<template>
        <a-flex :style="{ ...boxStyle }" justify="center" align="center" gap="large">
                <a-card title="轮次" :bordered="false" style="width: 30%; height: 150px">
                    <p style="text-align: center; font-size: 20px">{{epoch}}</p>
                </a-card>
                <a-card title="动作" :bordered="false" style="width: 30%; height: 150px;">
                    <p style="text-align: center; font-size: 20px">{{props.action}}</p>
                </a-card>
        </a-flex>
</template>

<script setup lang="ts">
import { reactive, ref, watchEffect } from 'vue';
import type { CSSProperties } from 'vue';
import type { FlexProps } from 'ant-design-vue';

const epoch = ref(0);

const props = defineProps({
  action: {
    type: String,
    required: true,
  },
});

watchEffect(() => {
  if (props.action === "正卧（一级）") {
    epoch.value = epoch.value + 1;
  }
});


const boxStyle: CSSProperties = {
        width: '100%',
        height: '200px',
        borderRadius: '6px',
        background: '#ececec',
};
</script>
  
  