import streamlit as st
import os
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import UnstructuredPDFLoader
from dotenv import load_dotenv
import time

load_dotenv()

groq_api_key = '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'

st.title("Virtual Diabetalogist 🧑🏻‍⚕️")
st.logo("C://Users//hp//Desktop//MyArchive//Virtual_Diabetalogist//logo.png")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only and act like a virtual doctor. Do not respond to any other question besides the provided context.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

def vector_embedding():
    persist_directory = "C://Users//hp//Desktop//MyArchive//Virtual_Diabetalogist//chroma_db"

    if os.path.exists(persist_directory):
        st.write("Loading vectors from disk...")
        st.session_state.vectors = Chroma(collection_name="local-rag",persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        st.write("Loaded vectors from disk.")
        return

    
    st.write("Creating new vectors...")
    
    st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    
    file_path = "C://Users//hp//Desktop//MyArchive//Virtual_Diabetalogist//data2.pdf"
    st.session_state.loader = UnstructuredPDFLoader(file_path)
    
    if not os.path.exists(file_path):
        st.write(f"File does not exist: {file_path}")
        return

    st.session_state.docs = st.session_state.loader.load()
    
    if not st.session_state.docs:
        st.write("No documents loaded.")
        return
    
    st.write(f"Loaded {len(st.session_state.docs)} documents.")
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:200])

    if not st.session_state.final_documents:
        st.write("No final documents after splitting.")
        return
    
    st.write(f"Created {len(st.session_state.final_documents)} document chunks.")
    
    for idx, doc in enumerate(st.session_state.final_documents):
        if 'id' not in doc.metadata or not doc.metadata['id']:
            doc.metadata['id'] = f"doc_{idx}"
    
    ids = [doc.metadata['id'] for doc in st.session_state.final_documents]
    
    if not ids or None in ids:
        st.write("IDs are not generated properly.")
        st.write(ids)
        return
    
    st.session_state.vectors = Chroma.from_documents(
        documents=st.session_state.final_documents,
        embedding=st.session_state.embeddings,
        collection_name="local-rag",
        persist_directory=persist_directory
    )
    
    st.session_state.vectors.persist()
    st.write("Vectors saved to disk.")

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if prompt1:
    if "vectors" not in st.session_state:
        st.write("Vectors not available. Please create embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(f"Response time: {time.process_time() - start}")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
