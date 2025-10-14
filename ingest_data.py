#!/usr/bin/env python
"""Script to ingest data files into the vectorstore.

This script reads file paths from data/files.txt and ingests them
into a FAISS vectorstore, then saves it for later use.

Optimizations:
- Uses caching to avoid reprocessing
- Batch processing for efficiency
- Parallel file loading
"""

import os
import argparse
from pathlib import Path
import shutil

from utils import logger, ingest_data_files
from store import create_faiss_vectorstore


def main(
    save: bool = True,
    test: bool = True,
    force_reload: bool = False,
):
    """Main function to ingest data.

    Args:
        save: Whether to save the vectorstore to disk
        test: Whether to run a test query
        force_reload: If True, ignore cache and reload all data
    """
    save_path = os.getenv(
        "VECTORSTORE_CACHE_PATH",
        Path(__file__).parent / "data" / "vectorstore",
    )

    # If force_reload is True, delete existing vectorstore
    if force_reload and save_path.exists():
        logger.info(
            f"Force reload enabled. Deleting existing vectorstore at {save_path}..."
        )
        shutil.rmtree(save_path)
        logger.info("Existing vectorstore deleted successfully!")

    cache_exists = save_path.exists()

    if cache_exists:
        logger.info(f"Loading existing vectorstore from {save_path}...")
        vectorstore = create_faiss_vectorstore(
            load_data=False, cache_path=str(save_path)
        )
    else:
        logger.info("Creating new vectorstore...")
        vectorstore = create_faiss_vectorstore(load_data=False)

        logger.info("Ingesting data from data/files.txt...")
        vectorstore = ingest_data_files(vectorstore)

        # Save the vectorstore
        if save:
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving vectorstore to {save_path}...")
            vectorstore.save_local(str(save_path))
            logger.info("Vectorstore saved successfully!")

    # Test the vectorstore with a sample query
    if test:
        logger.info("\n--- Testing vectorstore ---")
        test_query = "What is this document about?"
        results = vectorstore.similarity_search(test_query, k=3)

        logger.info(f"\nQuery: {test_query}")
        logger.info(f"Found {len(results)} results:\n")
        for i, doc in enumerate(results, 1):
            logger.info(f"Result {i}:")
            logger.info(f"  Content: {doc.page_content[:200]}...")
            logger.info(f"  Metadata: {doc.metadata}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data into vectorstore")
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save the vectorstore to disk"
    )
    parser.add_argument("--no-test", action="store_true", help="Don't run test query")
    parser.add_argument(
        "--force-reload", action="store_true", help="Ignore cache and reload all data"
    )

    args = parser.parse_args()

    main(
        save=not args.no_save,
        test=not args.no_test,
        force_reload=args.force_reload,
    )
