from flask import Flask, request, jsonify
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

groq_api_key = os.getenv('GROQ_API')

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only and act like a virtual doctor.
Do not respond to any other question besides the provided context. Do not use the word "context" or "provided text" or "given text"
at all instead always use phrases like "According to my knowledge" or "I think" etc.Always remain within the scope of the context, if a user asks a question outside the
scope of context simply reply "I'm sorry I have no knowledge about that".
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
""")

def vector_embedding():
    persist_directory = "C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//src//server_src//chroma_db"

    if os.path.exists(persist_directory):
        vectors = Chroma(collection_name="local-rag", persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        print("Vectors loaded from DB")
        return vectors

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    
    file_path = "C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//dataset//f_dataset.pdf"
    loader = UnstructuredPDFLoader(file_path)
    
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return 0

    docs = loader.load()
    
    if not docs:
        print("No documents loaded.")
        return 0
    
    print(f"Loaded {len(docs)} documents.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:200])

    if not final_documents:
        print("No final documents after splitting.")
        return 0
    
    print(f"Created {len(final_documents)} document chunks.")
    
    for idx, doc in enumerate(final_documents):
        if 'id' not in doc.metadata or not doc.metadata['id']:
            doc.metadata['id'] = f"doc_{idx}"
    
    ids = [doc.metadata['id'] for doc in final_documents]
    
    if not ids or None in ids:
        print("IDs are not generated properly.")
        print(ids)
        return 0
    
    vectors = Chroma.from_documents(
        documents=final_documents,
        embedding=embeddings,
        collection_name="local-rag",
        persist_directory=persist_directory
    )
    
    vectors.persist()
    print("Vectors saved to disk.")
    return vectors

vectors = vector_embedding()
print("Vector Store DB Is Ready")

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    prompt1 = data.get('prompt')
    
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        response_time = time.process_time() - start
        return jsonify({
            'response': response['answer'],
            'response_time': response_time
        })
    
    return jsonify({'error': 'Invalid input'}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
