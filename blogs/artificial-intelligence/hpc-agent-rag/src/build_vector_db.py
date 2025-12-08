import glob
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = DirectoryLoader("lumi-userguide/docs/", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

import chromadb

# For a persistent client
chroma_client = chromadb.PersistentClient(path="/root/.chroma_data") 

# Or for an in-memory client
# chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="LUMI_documentation")

ids = [f"doc_{i}" for i in range(len(chunks))] # Unique IDs for each chunk
texts = [chunk.page_content for chunk in chunks]

collection.add(
        documents=texts,
        ids=ids
    )
