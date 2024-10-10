from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from dotenv import load_dotenv
import time

load_dotenv()

app = Flask(__name__)
CORS(app)

groq_api_key = os.getenv('GROQ_API')

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")
#llm = Ollama(model="llama3:latest")

# Initialize conversation history and user data context
history = []
user_data_context = {}

# Define the chat prompt template
chat_template = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only (Never deviate from the 
topic or context even if the user insists to do so) and act like a virtual doctor, 
if the topic is unrelated to the context just reply with "I'm sorry I have no 
knowledge about that", gather information about user and respond the most accurate 
information according to that user information.

<CONTEXT>
{context}
</CONTEXT>
Questions: {input}
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
    global history, user_data_context
    data = request.get_json()
    print(data)
    user_input = data.get('prompt')
    
    if user_input:
        # Update conversation history
        history.append(("human", user_input))
        if len(history) > 5:
            history.pop(0)

        # Create context from history, retrieved documents, and user data context
        retrieved_context = get_retrieved_context(user_input)  # Function to retrieve relevant context
        formatted_context = "\n".join(f"{role}: {msg}" for role, msg in history) + "\n" + retrieved_context
        
        # Append user data context if it exists
        if user_data_context:
            formatted_context += f"\nUser data: {user_data_context}"

        # Format the prompt with history and context
        formatted_prompt = chat_template.format(context=formatted_context, input=user_input)

        # Use the LLM to get a response
        try:
            # Pass the formatted prompt to LLM
            llm_response = llm.invoke(formatted_prompt)
            ai_response = llm_response.content  # Access the content attribute directly
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            ai_response = "Sorry, there was an error processing your request."

        # Update history with the response
        history.append(("ai", ai_response))
        if len(history) > 5:
            history.pop(0)
        
        return jsonify({
            'response': ai_response,
        })
    
    return jsonify({'error': 'Invalid input'}), 400

# New endpoint to update user data context
@app.route('/update-context', methods=['POST'])
def update_context():
    global user_data_context
    data = request.get_json()
    
    # Update the user data context with the provided values
    user_data_context = {
        'bloodSugar': data.get('bloodSugar'),
        'heartRate': data.get('heartRate'),
        'age': data.get('age'),
        'glucoseLevels': data.get('glucoseLevels'),
        'hb1ac': data.get('hb1ac')
    }

    print("Updated user data context:", user_data_context)

    return jsonify({'message': 'User data context updated successfully'})

def get_retrieved_context(query):
    # Example function to retrieve context from documents
    document_chain = create_stuff_documents_chain(llm, chat_template)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    try:
        response = retrieval_chain.invoke({'input': query})
        return response.get('answer', '')  # Adjust based on actual response structure
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ''

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)
