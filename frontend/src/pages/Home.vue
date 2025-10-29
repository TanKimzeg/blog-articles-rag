<template>
  <div class="container">
    <header>
      <div class="title">Blog RAG 搜索</div>
    </header>

    <div class="panel">
      <div class="searchbar">
        <input v-model.trim="q" placeholder="搜索问题，例如：dropout与注意力权重" @keydown.enter="onSearch" />
        <select v-model="category">
          <option value="">全部分类</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
        <select v-model="tag">
          <option value="">全部标签</option>
          <option v-for="t in tags" :key="t" :value="t">{{ t }}</option>
        </select>
        <button :disabled="loading" @click="onSearch">{{ loading ? '搜索中...' : '搜索' }}</button>
      </div>
    </div>

    <div class="panel results">
      <div v-if="error" class="error">{{ error }}</div>
      <div v-else-if="!loading && results.length === 0" class="muted">暂无结果</div>
      <div v-else>
        <div v-for="(d,i) in results" :key="i" class="item">
          <div><strong>{{ i+1 }}. {{ d.metadata?.title || '未知标题' }}</strong></div>
          <div class="meta">
            <span v-if="d.metadata?.category || d.metadata?.categories?.[0]" class="badge">{{ d.metadata?.category || d.metadata?.categories?.[0] }}</span>
            <span v-for="t in d.metadata?.tags || []" :key="t" class="badge">{{ t }}</span>
          </div>
          <div class="muted" style="margin-top:8px; white-space:pre-wrap;">{{ (d.content || '').slice(0,300) }}</div>
          <div v-if="d.metadata?.parent_id || d.metadata?.file_id" style="margin-top:6px">
            <a href="#" @click.prevent="goView(d)">查看全文</a>
          </div>
        </div>
      </div>
    </div>

    <footer class="footer">Copyright © 2025 Blog RAG · <a href="https://github.com/TanKimzeg/blog-articles-rag" target="_blank">GitHub</a></footer>
  </div>
</template>

<script lang="ts" setup>
import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';

// API 基址：优先 VITE_API_BASE，否则使用同域 /api
const API_BASE = (import.meta as any).env?.VITE_API_BASE || '/api';
const apiUrl = (p: string) => `${API_BASE}${p}`;

const router = useRouter();

const q = ref('');
const category = ref('');
const tag = ref('');
const categories = ref<string[]>([]);
const tags = ref<string[]>([]);

const loading = ref(false);
const error = ref('');
const results = ref<any[]>([]);

async function loadFilters(){
  try{
    const [rc, rt] = await Promise.all([
      fetch(apiUrl('/meta/categories')),
      fetch(apiUrl('/meta/tags'))
    ]);
    const jc = await rc.json().catch(()=>({ data: { items: [] } }));
    const jt = await rt.json().catch(()=>({ data: { items: [] } }));
    const catItems = Array.isArray(jc?.data?.items) ? jc.data.items : [];
    const tagItems = Array.isArray(jt?.data?.items) ? jt.data.items : [];
    categories.value = catItems;
    tags.value = tagItems;
  }catch(err){
    console.warn('加载筛选项失败', err);
  }
}

async function onSearch(){
  if(!q.value.trim()) return;
  loading.value = true; error.value=''; results.value = [];
  const body = {
    query: q.value.trim(),
    topK: 10,
    page: 1,
    size: 10,
    filters: {
      categories: category.value ? [category.value] : undefined,
      tags: tag.value ? [tag.value] : undefined,
    },
    highlight: false,
  };
  try{
    const r = await fetch(apiUrl('/search'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    if(!r.ok) throw new Error('HTTP '+r.status);
    const resp = await r.json();
    const data = resp?.data;
    results.value = Array.isArray(data?.items) ? data.items : [];
  }catch(e:any){
    error.value = `请求失败：${e.message || e}`;
  }finally{
    loading.value = false;
  }
}

function goView(item:any){
  const id = item?.metadata?.parent_id || item?.metadata?.file_id || item?.metadata?.doc_id;
  if(id) router.push({ path: `/view/${id}` });
}

onMounted(()=>{ loadFilters(); });
</script>

<style scoped>
.container{ max-width:1000px; margin:0 auto; padding:24px; }
header{ display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
.title{ font-weight:600; letter-spacing:.4px; }
.panel{ background:var(--panel); border:1px solid rgba(255,255,255,.08); border-radius:12px; }
.searchbar{ display:flex; gap:12px; padding:12px; flex-wrap:wrap; }
input,select,button{ height:40px; border-radius:10px; border:1px solid rgba(255,255,255,.1); background:#0f1626; color:var(--text); padding:0 12px; }
input{ flex:1; min-width:220px; }
button{ background:linear-gradient(135deg,#4876ff,#7aa2ff); border:none; font-weight:600; cursor:pointer; }
button:disabled{ opacity:.6; cursor:not-allowed; }
.results{ margin-top:16px; padding:16px; }
.item{ padding:14px; border-bottom:1px solid rgba(255,255,255,.06); }
.meta{ color:var(--muted); font-size:12px; margin-top:6px; display:flex; gap:8px; flex-wrap:wrap; }
.badge{ padding:2px 8px; background:rgba(122,162,255,.18); border:1px solid rgba(122,162,255,.35); border-radius:999px; font-size:12px; }
.muted{ color:var(--muted); }
.footer{ text-align:center; color:var(--muted); font-size:12px; margin-top:24px; }
.error{ color:#ff8585; margin-top:8px; font-size:12px; }
</style>
