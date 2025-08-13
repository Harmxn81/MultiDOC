from langchain.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredImageLoader,
    # Add other loaders if needed
)

def load_document(file_path):
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
        docs = loader.load()
    elif ext == 'csv':
        loader = CSVLoader(file_path)
        docs = loader.load()
    elif ext in ['txt', 'md']:
        loader = TextLoader(file_path)
        docs = loader.load()
    elif ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
        # For images, you might need OCR loader or use UnstructuredImageLoader (requires unstructured lib)
        loader = UnstructuredImageLoader(file_path)
        docs = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return docs
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)

def chunk_docs(docs):
    texts = [doc.page_content for doc in docs]
    all_chunks = []
    for text in texts:
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
    return all_chunks
from langchain.embeddings import OpenAIEmbeddings

def embed_chunks(chunks):
    embedder = OpenAIEmbeddings()
    embeddings = embedder.embed_documents(chunks)
    return embeddings
from langchain.vectorstores import FAISS
def create_vectorstore(chunks, embeddings_model):
    # Create vectorstore from chunks and embeddings model
    vectorstore = FAISS.from_texts(chunks, embeddings_model)
    return vectorstore

def save_vectorstore(vectorstore, index_path="faiss_index"):
    vectorstore.save_local(index_path)

def load_vectorstore(index_path="faiss_index", embeddings_model=None):
    if embeddings_model is None:
        raise ValueError("Embeddings model required to load vectorstore.")
    return FAISS.load_local(index_path, embeddings_model)