<template>
    <a-table :columns="columns" :data-source="data" style="width: 100%; height: 100%;">
      <template v-slot:headerCell="{ column }">
        <div v-if="column.key === 'id'">ID</div>
        <div v-else-if="column.key === 'labels'">标签</div>
      </template>
      <template v-slot:bodyCell="{ column, record }">
        <div v-if="column.key === 'id'">{{ record.id }}</div>
        <div v-else-if="column.key === 'labels'">{{ record.labels }}</div>
      </template>
    </a-table>
  </template>
  
  <script lang="ts">
import { defineComponent } from 'vue';
import type { PropType } from 'vue';
  
  interface DataType {
    id: string;
    labels: string;
  }
  
  export default defineComponent({
    props: {
      data: {
        type: Array as PropType<DataType[]>,
        required: true
      }
    },
    setup(props) {
      const columns = [
        {
          title: 'ID',
          dataIndex: 'id',
          key: 'id'
        },
        {
          title: '标签',
          dataIndex: 'labels',
          key: 'labels'
        }
      ];
  
      return { columns };
    }
  });
  </script>
  
  