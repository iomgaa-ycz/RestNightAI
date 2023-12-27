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
import { watchEffect } from 'vue';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid';

interface DelayLoading {
  delay: number;
}

const iconLoading = ref<boolean | DelayLoading>(false);
const level = ref<number>(0);
const Countdown = ref<number>(120);
const id = ref<string>(''); // Reactive variable to store the UUID
const action = ref<string>(''); // Reactive variable to store the action

// Define the props and emits
const props = defineProps();
const emit = defineEmits(['updateLevel', 'updateAction', 'updateCountdown']);

const enterIconLoading = async () => {
  if (iconLoading.value || Countdown.value != 120 || level.value != 0) {
    return;
  }
  Countdown.value = Countdown.value - 1;
  emit('updateCountdown', Countdown.value); // Emit the updateCountdown event
  id.value = uuidv4();
  try {
    const time = new Date().toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' }); // Convert to Beijing time
    const response = await axios.post('/api/begin_collect', {
      Time: time,
      Action: action.value,
      ID: id.value // Send the UUID to the server
    });

    if (response.status === 200) {
      iconLoading.value = true;
      level.value = 1; // Increment level by 1
      console.log('level updated to: ', level.value);
      emit('updateLevel', level.value); // Emit the updateLevel event

    }
  } catch (error) {
    console.error(error);
  }
};

const enternext = async () => {
  iconLoading.value = false;
  if (level.value == 1) {
    level.value = 2; // Increment level by 2
    try {
      const time = new Date().toLocaleString('zh-CN', { timeZone: 'Asia/Shanghai' }); // Convert to Beijing time
      const response = await axios.post('/api/finish_collect', {
        Time: time,
        Action: action.value,
        ID: id.value // Send the UUID to the server
      });
      console.log(response.data);
    } catch (error) {
      console.error(error);
    } 
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

// Update current and percent based on level prop
watchEffect(async () => {
  if (level.value === 0) {
    try {
      const response = await axios.get('/api/collect_action');
      action.value = response.data.action;
      emit('updateAction', action.value); // Emit the updateAction event
    } catch (error) {
      console.error(error);
    }
  } 
});

// Timer to decrement Countdown value every second until it reaches 0
const timer = setInterval(() => {
  if (Countdown.value !== 120) {
    Countdown.value -= 1;
    emit('updateCountdown', Countdown.value); // Emit the updateCountdown event
    if (Countdown.value === 0) {
      Countdown.value = 120;
      emit('updateCountdown', Countdown.value); // Emit the updateCountdown event
    }
  }
}, 1000);

// Export the level and uuid variables
const { level: exposedLevel } = toRefs({ level });
const { uuid: exposedUuid } = toRefs({ uuid: id });
</script>