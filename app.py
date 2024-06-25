import streamlit as st
import cohere
import uuid
import hnswlib
import requests
from io import BytesIO
import fitz
from typing import List, Dict

# Initialize Cohere client
co = cohere.Client("Cohere API Key")

class Vectorstore:
    def __init__(self, raw_documents: List[Dict[str, str]]):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        for raw_document in self.raw_documents:
            pdf_url = raw_document["url"]
            pdf_content = self.download_pdf(pdf_url)
            chunks = self.chunk_pdf_content(pdf_content)
            for chunk in chunks:
                self.docs.append({
                    "title": raw_document["title"],
                    "text": chunk,
                    "url": pdf_url,
                })

    def download_pdf(self, url: str) -> str:
        response = requests.get(url)
        response.raise_for_status()
        
        pdf_stream = BytesIO(response.content)
        pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")

        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            text += page.get_text()

        return text

    def chunk_pdf_content(self, content: str) -> List[str]:
        chunk_size = 2000
        return [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

    def embed(self) -> None:
        batch_size = 90
        self.docs_len = len(self.docs)
        for i in range(0, self.docs_len, batch_size):
            batch = self.docs[i: min(i + batch_size, self.docs_len)]
            texts = [item["text"] for item in batch]
            docs_embs_batch = co.embed(
                texts=texts, model="embed-english-v3.0", input_type="search_document"
            ).embeddings
            self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        self.idx = hnswlib.Index(space="ip", dim=1024)
        self.idx.init_index(max_elements=self.docs_len, ef_construction=512, M=64)
        self.idx.add_items(self.docs_embs, list(range(len(self.docs_embs))))

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        query_emb = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings

        doc_ids = self.idx.knn_query(query_emb, k=self.retrieve_top_k)[0][0]

        rank_fields = ["title", "text"]

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids]

        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )

        doc_ids_reranked = [doc_ids[result.index] for result in rerank_results.results]

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append({
                "title": self.docs[doc_id]["title"],
                "text": self.docs[doc_id]["text"],
                "url": self.docs[doc_id]["url"],
            })

        return docs_retrieved

class Chatbot:
    def __init__(self, vectorstore: Vectorstore):
        self.vectorstore = vectorstore
        self.conversation_id = str(uuid.uuid4())

    def respond(self, message: str):
        response = co.chat(
            message=message,
            model="command-r",
            search_queries_only=True
        )

        documents = []
        if response.search_queries:
            for query in response.search_queries:
                docs = self.vectorstore.retrieve(query.text)
                documents.extend(docs)

            if documents:
                response = co.chat_stream(
                    message=message,
                    model="command-r-plus",
                    documents=documents,
                    conversation_id=self.conversation_id,
                )
            else:
                response = co.chat_stream(
                    message=message,
                    model="command-r-plus",
                    conversation_id=self.conversation_id,
                )

        chat_response = ""
        citations = []
        cited_documents = []

        for event in response:
            if event.event_type == "text-generation":
                chat_response += event.text
            elif event.event_type == "citation-generation":
                citations.extend(event.citations)
            elif event.event_type == "stream-end":
                cited_documents = event.response.documents

        return chat_response, citations, cited_documents

# Streamlit UI
def main():
    st.title("PDF Document Chatbot")

    # Initialize raw documents
    raw_documents = [
        {"title": "USAID Report", "url": "https://pdf.usaid.gov/pdf_docs/PA00TBCT.pdf"}
    ]

    # Create an instance of the Vectorstore class
    vectorstore = Vectorstore(raw_documents)
    chatbot = Chatbot(vectorstore)

    st.write("Ask me anything about the provided documents:")

    user_input = st.text_input("Your question:")
    if st.button("Send"):
        if user_input:
            response, citations, documents = chatbot.respond(user_input)
            st.write("Chatbot response:")
            st.write(response)

            if citations:
                st.write("Citations:")
                for citation in citations:
                    st.write(citation)

            if documents:
                st.write("Referenced Documents:")
                for document in documents:
                    st.write({
                        'id': document.get('id', 'N/A'),
                        'snippet': document.get('snippet', '')[:400] + '...',
                        'title': document.get('title', 'N/A'),
                        'url': document.get('url', 'N/A')
                    })

if __name__ == "__main__":
    main()
