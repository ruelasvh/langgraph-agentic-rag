from store import create_faiss_vectorstore
from langchain_core.documents import Document


def test_add_and_query_empty_store():
    vs = create_faiss_vectorstore()
    vs.add_documents(
        [
            Document(
                page_content="This is a test document about revenue growth.",
                metadata={
                    "title": "t1",
                },
            )
        ]
    )
    results = vs.similarity_search(query="revenue")
    assert len(results) >= 1
