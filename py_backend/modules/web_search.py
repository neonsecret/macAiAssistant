import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def extract_user_text(html_content):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Limit processing to the body of the document
    body = soup.body
    if body is None:
        return set()

    unique_sentences = set()

    allowed_tags = ["p"]  # ['a', 'p', 'span']
    for tag in body.find_all(allowed_tags):
        # Extract only the user-visible text (ignores tag attributes like links)
        tag_text = tag.get_text(" ", strip=True)
        if tag_text:
            # Split text into sentences.
            # The regex splits after a period, exclamation, or question mark followed by whitespace.
            sentences = re.split(r'(?<=[.!?])\s+', tag_text)
            for sentence in sentences:
                cleaned = sentence.strip()
                cleaned = re.sub(r'\[.*?\]', '', cleaned)
                if cleaned:
                    unique_sentences.add(cleaned)

    return unique_sentences


def search_topic_RAG(query: str):
    results = DDGS().text(query, max_results=10)
    total_set = set()
    for result in results:
        html_content = requests.get(result["href"]).content.decode()
        page_content_set = extract_user_text(html_content)
        page_content_set.add(result["body"])
        total_set = total_set | page_content_set

    documents = list(total_set)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = embedder.encode(documents)
    doc_embeddings = np.array(doc_embeddings).astype("float32")

    # Build FAISS index using L2 distance (or choose your preferred metric)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings)

    # --- Retrieval Query ---
    query_embedding = embedder.encode([query]).astype("float32")
    k = 3  # Retrieve top-2 documents
    distances, indices = index.search(query_embedding, k)
    retrieved_docs = [documents[i] for i in indices[0]]
    return " ".join(retrieved_docs)
    # print(retrieved_docs)


if __name__ == '__main__':
    search_topic_RAG("How old is adrien brody?")
