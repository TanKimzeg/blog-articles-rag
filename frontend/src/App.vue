<template>
  <router-view />
</template>

<script lang="ts" setup>
// 空壳组件，实际页面由路由渲染
</script>

<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
:root { --bg:#0b1220; --panel:#121a2b; --muted:#9aa4b2; --text:#e6edf7; --acc:#7aa2ff; }
html,body,#app{ height:100%; }
body{ margin:0; background:var(--bg); color:var(--text); font-family: Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial,Apple Color Emoji,Segoe UI Emoji; }
a{ color:var(--acc); text-decoration:none; }
</style>
