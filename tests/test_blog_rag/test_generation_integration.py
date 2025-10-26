from blog_rag.rag_modules import GenerationIntegrationModule


def test_generation_module_init_with_env():
    # 只验证能构造实例（不会真正调用接口）
    gim = GenerationIntegrationModule(model_name="deepseek-chat", temperature=0.0, max_tokens=16)
    assert gim is not None
