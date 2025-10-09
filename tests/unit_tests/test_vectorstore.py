from store import create_faiss_vectorstore


def test_add_and_query_empty_store():
    vs = create_faiss_vectorstore()
    vs.add_documents(
        [{"title": "t1", "text": "This is a test document about revenue growth."}]
    )
    results = vs.query("revenue")
    assert len(results) >= 1
