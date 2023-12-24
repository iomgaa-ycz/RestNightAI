<template>
    <a-layout class="layer">
      <a-layout-header class="header">
        <TheHead />
      </a-layout-header>
      <a-layout>
        <a-layout-sider width="200" style="background: #fff">
          <LeftSide />
        </a-layout-sider>
        <a-layout style="padding: 0 24px 24px">
        </a-layout>
      </a-layout>
    </a-layout>
  </template>
  <script lang="ts" setup>
  import { ref } from 'vue';
  import { UserOutlined, LaptopOutlined, NotificationOutlined } from '@ant-design/icons-vue';
  import TheHead from '@/layer/TheHead.vue';
  import LeftSide from '@/layer/LeftSide.vue';
  const selectedKeys1 = ref<string[]>(['2']);
  const selectedKeys2 = ref<string[]>(['1']);
  const openKeys = ref<string[]>(['sub1']);
  </script>
  <style scoped>
    .layer {
      height: 100vh;
    }
  </style>
