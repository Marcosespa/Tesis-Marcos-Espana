def test_chunking_import():
    from src.process.chunking import build_chunks
    assert callable(build_chunks)
