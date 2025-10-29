import { createRouter, createWebHistory } from 'vue-router';
import Home from './pages/Home.vue';

const routes = [
  { path: '/', component: Home },
  { path: '/view/:id', component: () => import('@/view/docView/Index.vue') }
];

export const router = createRouter({
  history: createWebHistory(),
  routes,
  scrollBehavior() { return { top: 0 }; }
});
