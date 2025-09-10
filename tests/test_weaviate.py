def test_weaviate_client_import():
    from src.index.weaviate_client import get_client
    assert get_client is not None
