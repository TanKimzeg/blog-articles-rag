<template>
  <div class="doc-container">
    <header class="doc-header">
      <a class="back" href="#" @click.prevent="$router.back()">← 返回</a>
      <div class="title">{{ title || '文档' }}</div>
    </header>

    <div class="panel doc-body">
      <div v-if="error" class="error">{{ error }}</div>
      <div v-else-if="loading" class="muted">加载中...</div>
      <article v-else class="markdown" v-html="html"></article>
    </div>
  </div>
</template>

<script lang="ts" setup>
import MarkdownIt from 'markdown-it';
import { onMounted, ref, watch } from 'vue';
import { useRoute } from 'vue-router';

const md = new MarkdownIt({ html: false, linkify: true, breaks: true });

// API 基址：优先 VITE_API_BASE，否则使用同域 /api
const API_BASE = (import.meta as any).env?.VITE_API_BASE || '/api';
const apiUrl = (p: string) => `${API_BASE}${p}`;

const route = useRoute();
const loading = ref(false);
const error = ref('');
const title = ref('');
const html = ref('');

async function fetchDoc(id: string){
  if(!id) return;
  loading.value = true; error.value='';
  try{
    const r = await fetch(apiUrl(`/docs/${id}`));
    if(!r.ok) throw new Error('HTTP '+r.status);
    const j = await r.json();
    const data = j?.data ?? j;
    title.value = data?.metadata?.title || '文档';
    html.value = md.render(String(data?.content || ''));
  }catch(e:any){
    error.value = `加载文档失败：${e.message || e}`;
  }finally{
    loading.value = false;
  }
}

onMounted(()=>{
  fetchDoc(String(route.params.id||''));
});

watch(() => route.params.id, (id) => { if(id) fetchDoc(String(id)); });
</script>

<style scoped>
.doc-container{ max-width:900px; margin:0 auto; padding:24px; }
.doc-header{ display:flex; align-items:center; gap:12px; margin-bottom:12px; }
.back{ color:#7aa2ff; }
.title{ font-weight:600; }
.panel{ background:var(--panel); border:1px solid rgba(255,255,255,.08); border-radius:12px; }
.doc-body{ padding:16px; }
.muted{ color:var(--muted); }
.error{ color:#ff8585; }

/* 重要：v-html 渲染的节点没有 scope attribute，需要用 :deep() 才能命中 */
:deep(.markdown){ box-sizing:border-box; overflow-wrap:anywhere; word-break:break-word; }
:deep(.markdown :is(h1,h2,h3,h4)){ margin: 1em 0 .5em; }
:deep(.markdown pre){ background:#0f1626; padding:12px; border-radius:8px; overflow:auto; }
:deep(.markdown pre code){ background:transparent !important; padding:0 !important; border-radius:0; }
:deep(.markdown img){ max-width:100% !important; height:auto !important; display:block; margin:8px auto; box-sizing:border-box; }
:deep(.markdown figure){ margin:8px auto; }
:deep(.markdown table){ width:100%; display:block; overflow-x:auto; }
:deep(.markdown p){ line-height:1.75; }
:deep(.markdown blockquote){ border-left:4px solid var(--acc); padding-left:12px; color:var(--muted); margin:1em 0; }
:deep(.markdown code){ background:rgba(255,255,255,.06); padding:.2em .35em; border-radius:4px; }
</style>
