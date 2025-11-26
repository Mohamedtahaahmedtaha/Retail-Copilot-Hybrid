import os
import glob
from rank_bm25 import BM25Okapi
from dspy.primitives.module import Module

class Document:
    def __init__(self, text, metadata=None):
        self.text = text
        self.metadata = metadata if metadata is not None else {}

DOCS_DIR = "docs"

class RagRetriever(Module):
    """
    A RAG (Retrieval-Augmented Generation) tool for retrieving relevant document chunks
    from the local markdown files (docs/). Uses BM25 for indexing and search.
    """
    
    def __init__(self):
        super().__init__()
        self.documents = []
        self._load_and_chunk_docs()
        self.bm25 = None
        
        if self.documents:
            tokenized_corpus = [doc.text.split(" ") for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
        else:
            print(f"Warning: No documents loaded from {DOCS_DIR}. RAG will return empty results.")

    def _load_and_chunk_docs(self):
        """
        Loads all markdown files from the docs directory and chunks them into Documents.
        """
        file_paths = glob.glob(os.path.join(DOCS_DIR, '*.md'))
        
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # split documents to chunks
                chunks = content.split('\n')
                
                for chunk in chunks:
                    if chunk.strip():
                        self.documents.append(
                            Document(text=chunk.strip(), metadata={"source": os.path.basename(file_path)})
                        )
            except Exception as e:
                print(f"Error loading document {file_path}: {e}")

    def run(self, question, k=5):
        """
        Retrieves the top k most relevant document chunks based on the question.
        
        """
        if not self.bm25:
            return []
            
        tokenized_query = question.split(" ")
        
        # calcukatuib score BM25
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # top score
        top_n_indices = doc_scores.argsort()[-k:][::-1]
        
        retrieved_docs = [self.documents[i] for i in top_n_indices]
        
        return retrieved_docs

if __name__ == '__main__':
    
    retriever = RagRetriever()
    question = "What is the policy for beverages return?"
    docs = retriever.run(question)
    print(f"Retrieved {len(docs)} documents:")
    for doc in docs:
        print(f"  [{doc.metadata['source']}] {doc.text}")