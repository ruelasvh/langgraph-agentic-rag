"""Utility functions for the agent."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
from langgraph.graph import MessagesState
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_file(file_path: str) -> List:
    """Load a single file based on its extension.

    Args:
        file_path: Path to the file to load

    Returns:
        List of Document objects

    Raises:
        FileNotFoundError: If the file doesn't exist
        ImportError: If required loader dependencies are missing
    """
    # Lazy imports to avoid loading heavy dependencies until needed
    from langchain_community.document_loaders import (
        PyPDFLoader,
        UnstructuredFileLoader,
    )

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine loader based on file extension
    extension = path.suffix.lower()

    try:
        if extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            # Use UnstructuredFileLoader for other file types
            loader = UnstructuredFileLoader(file_path)

        return loader.load()
    except ImportError as e:
        logger.error(f"Missing dependency for {extension} files: {e}")
        raise


def _load_and_split_file(
    file_path: str,
    text_splitter,
) -> tuple[str, List, Optional[Exception]]:
    """Helper function to load and split a single file.

    Args:
        file_path: Path to the file to load
        text_splitter: Text splitter instance

    Returns:
        Tuple of (file_path, split_documents, error)
    """
    try:
        documents = load_file(file_path)
        split_docs = text_splitter.split_documents(documents)
        return file_path, split_docs, None
    except Exception as e:
        return file_path, [], e


def ingest_files_from_list(
    files_list_path: str,
    vectorstore: FAISS,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    batch_size: int = 100,
    max_workers: int = 4,
) -> FAISS:
    """Ingest files listed in a text file into a FAISS vectorstore.

    Optimizations:
    - Parallel file loading using ThreadPoolExecutor
    - Batch insertion of documents to reduce overhead
    - Lazy import of text splitter
    - Better error handling and logging

    Args:
        files_list_path: Path to the text file containing file paths (one per line)
        vectorstore: The FAISS vectorstore to add documents to
        chunk_size: Size of text chunks for splitting
        chunk_overlap: Overlap between chunks
        batch_size: Number of documents to add at once (improves performance)
        max_workers: Maximum number of parallel file loading workers

    Returns:
        Updated FAISS vectorstore with ingested documents
    """
    # Lazy import to avoid loading until needed
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    files_list_path_obj = Path(files_list_path)

    if not files_list_path_obj.exists():
        raise FileNotFoundError(f"Files list not found: {files_list_path}")

    # Read the list of files
    with open(files_list_path, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]

    if not file_paths:
        logger.warning("No files found in the list.")
        return vectorstore

    # Initialize text splitter (reuse for all files)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    all_documents = []
    total_files = len(file_paths)

    logger.info(f"Loading {total_files} files with {max_workers} workers...")

    # Parallel file loading for better performance
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all file loading tasks
        future_to_path = {
            executor.submit(_load_and_split_file, file_path, text_splitter): file_path
            for file_path in file_paths
        }

        # Process completed tasks as they finish
        for i, future in enumerate(as_completed(future_to_path), 1):
            file_path, split_docs, error = future.result()

            if error:
                logger.error(f"Error loading {file_path}: {error}")
            else:
                all_documents.extend(split_docs)
                logger.info(
                    f"[{i}/{total_files}] Loaded {len(split_docs)} chunks from {Path(file_path).name}"
                )

    if not all_documents:
        logger.warning("No documents were loaded.")
        return vectorstore

    # Add documents in batches for better performance
    total_docs = len(all_documents)
    logger.info(
        f"Adding {total_docs} document chunks to vectorstore in batches of {batch_size}..."
    )

    for i in range(0, total_docs, batch_size):
        batch = all_documents[i : i + batch_size]
        vectorstore.add_documents(batch)
        logger.info(
            f"Added batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}"
        )

    logger.info("Documents successfully added to vectorstore!")
    return vectorstore


def ingest_data_files(vectorstore: FAISS) -> FAISS:
    """Convenience function to ingest files from data/files.txt.

    Args:
        vectorstore: The FAISS vectorstore to add documents to

    Returns:
        Updated FAISS vectorstore with ingested documents
    """
    # Assuming this is called from the project root or we need to find it
    project_root = Path(__file__).parent.parent
    files_list_path = project_root / "data" / "files.txt"

    return ingest_files_from_list(str(files_list_path), vectorstore)


def get_latest_messages(
    state: MessagesState, return_context: bool = False
) -> tuple[str | None, str | None]:
    """Extract the most recent user question and optionally the tool context from message history.

    Args:
        state: The current message state
        return_context: If True, also return the most recent tool message content

    Returns:
        A tuple of (question, context) where:
        - question: The most recent human/user message content
        - context: The most recent tool message content (only if return_context=True, otherwise None)
    """
    question = None
    context = None

    for msg in reversed(state["messages"]):
        if hasattr(msg, "type"):
            if msg.type == "human" and question is None:
                question = msg.content
            elif msg.type == "tool" and return_context and context is None:
                context = msg.content
        elif isinstance(msg, dict):
            if msg.get("role") == "user" and question is None:
                question = msg.get("content")
            elif msg.get("role") == "tool" and return_context and context is None:
                context = msg.get("content")

        # Early exit if we found everything we need
        if question is not None and (not return_context or context is not None):
            break

    return question, context
