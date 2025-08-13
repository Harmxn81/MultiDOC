from utils import (
    load_document,
    chunk_docs,
    embed_chunks,
    create_vectorstore,
    save_vectorstore,
    load_vectorstore,
)
from langchain.embeddings import OpenAIEmbeddings
import os

def main():
    file_path = input("Enter file path: ").strip()
    index_dir = "faiss_index"

    print("Loading document...")
    docs = load_document(file_path)

    print("Splitting document into chunks...")
    chunks = chunk_docs(docs)
    print(f"Total chunks: {len(chunks)}")

    embeddings_model = OpenAIEmbeddings()

    if os.path.exists(index_dir):
        print(f"Loading existing vectorstore from '{index_dir}'...")
        vectorstore = load_vectorstore(index_dir, embeddings_model)
    else:
        print("Creating new vectorstore and embedding chunks...")
        vectorstore = create_vectorstore(chunks, embeddings_model)
        print("Saving vectorstore...")
        save_vectorstore(vectorstore, index_dir)

    while True:
        query = input("\nEnter query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        results = vectorstore.similarity_search(query, k=3)
        print(f"Top results for query: '{query}'")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:\n{doc.page_content}\n{'-'*40}")

if __name__ == "__main__":
    main()