from fastapi.testclient import TestClient


def test_meta_tags(client: TestClient):
    resp = client.get("/meta/tags")
    assert resp.status_code == 200
    data = resp.json().get("data", {})
    assert isinstance(data.get("items", []), list)


def test_meta_categories(client: TestClient):
    resp = client.get("/meta/categories")
    assert resp.status_code == 200
    data = resp.json().get("data", {})
    assert isinstance(data.get("items", []), list)


def test_search(client: TestClient):
    payload = {"query": "dropout", "topK": 5, "page": 1, "size": 5}
    resp = client.post("/search", json=payload)
    assert resp.status_code == 200
    data = resp.json().get("data", {})
    items = data.get("items", [])
    assert isinstance(items, list)
    assert len(items) <= 5


def test_get_doc(client: TestClient):
    # Perform a search to obtain a doc id
    payload = {"query": "dropout", "topK": 1, "page": 1, "size": 1}
    search_resp = client.post("/search", json=payload)
    assert search_resp.status_code == 200
    search_data = search_resp.json().get("data", {})
    items = search_data.get("items", [])
    if not items:
        return
    meta = items[0].get("metadata", {})
    doc_id = meta.get("doc_id") or meta.get("parent_id") or meta.get("file_id")

    doc_resp = client.get(f"/docs/{doc_id}")
    assert doc_resp.status_code == 200
    doc_data = doc_resp.json().get("data", {})
    assert "content" in doc_data
    assert "metadata" in doc_data
    assert "path" in doc_data
