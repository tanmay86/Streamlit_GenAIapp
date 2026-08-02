import time

import streamlit as st
from loaders.pdf_loader import load_pdf
from embeddings.embedder import Embedder
from vectorstore.vectorstore import VectorStore
from llm.llm_client import LLMClient
from utils.utils import read_pdf  # Example utility function
import os

def main():
    
    def stream_lines(llm_raw_response):
   
    
        for line in llm_raw_response.split(" "):
            yield line + " "
            time.sleep(0.1)
    
    st.title("Pdf Gen AI App")
    
    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
         with st.spinner("Reading document and generating embeddings..."):
        # Load PDF and extract text
            pdf_text = read_pdf(uploaded_file.getvalue())
        #st.write("Extracted Text:")
       # st.write(pdf_text)
        
        # Generate embeddings
            embedder = Embedder("")
            embeddings,Chunks = embedder.generate_embeddings(pdf_text)
        
        # Store embeddings in vector database
            vector_store = VectorStore("")
            retriever= vector_store.add_embedding(embeddings,Chunks)
        
        # Initialize LLM client
            llm_client = LLMClient("llama-3.2-3b-instruct")
        
        # User input for querying the LLM
    user_query = st.text_input("Ask a question:")
        
        
    if user_query:
            with st.spinner("Thinking... Fetching documents and generating answer..."):
        
   
                response = llm_client.query(user_query,retriever)
                st.write("LLM Response:")
        
            only_answer = response["answer"]
            st.write(stream_lines(only_answer))

if __name__ == "__main__":
    main()