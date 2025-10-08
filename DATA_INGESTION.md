# Data Ingestion Guide

This guide explains how to ingest data files into the FAISS vectorstore for the agentic RAG system.

## How It Works

The system reads file paths from `data/files.txt` (one file path per line), loads the documents, splits them into chunks, and adds them to the FAISS vectorstore.

## File Format

Edit `data/files.txt` to include the paths to files you want to ingest:

```txt
/path/to/your/document.pdf
/path/to/another/document.pdf
```

## Supported File Types

- **PDF files** (`.pdf`) - Uses PyPDFLoader
- **Other formats** - Uses UnstructuredFileLoader (may require additional dependencies)

## Usage

### Option 1: Automatic Loading (Recommended for Development)

The graph automatically loads data when initialized if `load_data=True` is set in `create_faiss_vectorstore()`:

```python
# In graph.py
vector_store = create_faiss_vectorstore(load_data=True)
```

This will load data every time the graph starts, which is useful for development.

### Option 2: Manual Ingestion Script

Run the ingestion script to load data and save the vectorstore:

```bash
python ingest_data.py
```

This will:
1. Create a new vectorstore
2. Load files from `data/files.txt`
3. Save the vectorstore to `data/vectorstore/`
4. Run a test query to verify the data

### Option 3: Programmatic Usage

Use the utility functions directly in your code:

```python
from src.agent.vectorstore import create_faiss_vectorstore
from src.agent.utils import ingest_data_files

# Create vectorstore
vectorstore = create_faiss_vectorstore(load_data=False)

# Ingest data
vectorstore = ingest_data_files(vectorstore)

# Or ingest from a custom file list
from src.agent.utils import ingest_files_from_list
vectorstore = ingest_files_from_list(
    "path/to/custom_files.txt",
    vectorstore,
    chunk_size=1000,
    chunk_overlap=200
)
```

## Configuration

You can customize the text splitting parameters:

```python
from src.agent.utils import ingest_files_from_list

vectorstore = ingest_files_from_list(
    files_list_path="data/files.txt",
    vectorstore=vectorstore,
    chunk_size=500,      # Smaller chunks
    chunk_overlap=100    # Less overlap
)
```

## Troubleshooting

### Missing Dependencies

If you encounter import errors, install the required packages:

```bash
pip install pypdf unstructured
```

### File Not Found

Ensure:
- File paths in `data/files.txt` are absolute paths or correct relative paths
- Files actually exist at the specified locations
- You have read permissions for the files

### Memory Issues

For large documents:
- Reduce `chunk_size` (default: 1000)
- Process files in smaller batches
- Consider using a persistent vectorstore instead of in-memory

## Example Workflow

1. **Add file paths to data/files.txt**:
   ```bash
   echo "/path/to/your/document.pdf" >> data/files.txt
   ```

2. **Run the ingestion**:
   ```bash
   python ingest_data.py
   ```

3. **Test with the graph**:
   ```bash
   langgraph dev
   ```

4. **Query in the LangGraph Studio UI** or via API:
   ```python
   response = graph.invoke({
       "messages": [{"role": "user", "content": "What is in the document?"}]
   })
   ```
