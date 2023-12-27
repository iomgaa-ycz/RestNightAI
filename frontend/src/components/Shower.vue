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
import { reactive, ref } from 'vue';
import type { CSSProperties } from 'vue';
import type { FlexProps } from 'ant-design-vue';

const epoch = ref(0);

const props = defineProps({
  action: {
    type: String,
    required: true,
  },
});

const justifyOptions = reactive<FlexProps['justify'][]>([
        'flex-start',
        'center',
        'flex-end',
        'space-between',
        'space-around',
        'space-evenly',
]);

const alignOptions = reactive<FlexProps['align'][]>(['flex-start', 'center', 'flex-end']);
const justify = ref(justifyOptions[0]);
const alignItems = ref(alignOptions[0]);
const boxStyle: CSSProperties = {
        width: '100%',
        height: '200px',
        borderRadius: '6px',
        background: '#ececec',
};
</script>
  
  