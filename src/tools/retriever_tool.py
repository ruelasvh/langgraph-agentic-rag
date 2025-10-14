"""Retriever tool factory with lazy loading support."""

import os
from pathlib import Path
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts.prompt import PromptTemplate
from store import create_faiss_vectorstore


_retriever_tool = None


def get_retriever_tool():
    """Lazy load and return the retriever tool.

    Uses cached vectorstore if available for better performance.
    Set USE_VECTORSTORE_CACHE=false to disable caching.
    """
    global _retriever_tool

    if _retriever_tool is not None:
        return _retriever_tool

    # Use cached vectorstore if available for better performance
    # Set USE_VECTORSTORE_CACHE=false to disable caching
    cache_path = os.getenv(
        "VECTORSTORE_CACHE_PATH",
        Path(__file__).parent.parent.parent / "data" / "vectorstore",
    )
    use_cache = os.getenv("USE_VECTORSTORE_CACHE", "true").lower() != "false"

    if use_cache and Path(cache_path).exists():
        vector_store = create_faiss_vectorstore(load_data=False, cache_path=cache_path)
    else:
        vector_store = create_faiss_vectorstore(load_data=True)

    # Create retriever tool
    retriever = vector_store.as_retriever()
    doc_prompt = PromptTemplate.from_template(
        "<context>\n{page_content}\n\n<meta>\nsource: {source}\npage: {page}\n</meta>\n</context>"
    )
    _retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and return information about documents.",
        document_prompt=doc_prompt,
    )

    return _retriever_tool
