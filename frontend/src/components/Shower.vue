<template>
        <a-flex :style="{ ...boxStyle }" justify="center" align="center" gap="large">
                <a-card title="轮次" :bordered="false" style="width: 20%; height: 150px">
                    <p style="text-align: center; font-size: 20px">{{epoch}}</p>
                </a-card>
                <a-card title="动作" :bordered="false" style="width: 20%; height: 150px;">
                    <p style="text-align: center; font-size: 20px">{{props.action}}</p>
                </a-card>
                <a-card title="倒计时" :bordered="false" style="width: 20%; height: 150px;">
                    <p style="text-align: center; font-size: 20px">{{props.Countdown}}</p>
                </a-card>
                <a-card title="数据量" :bordered="false" style="width: 20%; height: 150px;">
                    <p style="text-align: center; font-size: 20px">{{DatabaseNum}}</p>
                </a-card>
        </a-flex>
</template>

<script setup lang="ts">
import { reactive, ref, watchEffect } from 'vue';
import type { CSSProperties } from 'vue';
import type { FlexProps } from 'ant-design-vue';

const epoch = ref(0);
const DatabaseNum = ref("0");

const props = defineProps({
  action: {
    type: String,
    required: true,
  },
  Countdown: {
    type: Number,
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

// Send GET request every 5 seconds
setInterval(() => {
  fetch('/api/get_database_key_number')
    .then(response => response.json())
    .then(data => {
      console.log(data.number)
      DatabaseNum.value = data.number;
    })
    .catch(error => {
      console.error('Error:', error);
    });
}, 5000);
</script>
