from flask import Flask, request, jsonify, send_file, make_response
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
import torch
from PIL import Image
import pandas as pd
import numpy as np
import torchvision.transforms as transforms  
import torch.nn as nn 
import torch.nn.functional as F
import base64
from io import BytesIO
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from fpdf import FPDF 
import io
import re

load_dotenv()

app = Flask(__name__)
CORS(app)

groq_api_key = os.getenv('GROQ_API')

llm = ChatGroq(groq_api_key=groq_api_key, 
                model_name="llama3-8b-8192", 
                temperature=0.5, 
                verbose=True, n=1, 
                max_retries=2)
#llm = OllamaLLM(model="llama3:latest", temperature=0.5, verbose=True, max_retries=2)

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



bnb_quantization = {
    "bnb_config": {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "torch.bfloat16"  
    },
    "base_model": "Meta-Llama-3.1-8B-Instruct",
    "tokenizer": {
        "load_tokenizer": "AutoTokenizer.from_pretrained(base_model)"
    },
    "base_model_bnb_4b": {
        "load_model": "AutoModelForCausalLM.from_pretrained(base_model, quantization_config=bnb_config, device_map='auto')"
    }
}

target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]

model_args = {
    "Llama3@latest": {
        "model": {
            "embed_tokens": {
                "type": "Embedding",
                "params": {
                    "num_embeddings": 128256,
                    "embedding_dim": 4096
                }
            },
            "layers": [
                {
                    "LlamaDecoderLayer": {
                        "self_attn": {
                            "LlamaSdpaAttention": {
                                "q_proj": {
                                    "type": "Linear4bit",
                                    "params": {
                                        "in_features": 4096,
                                        "out_features": 4096,
                                        "bias": False
                                    }
                                },
                                "k_proj": {
                                    "type": "Linear4bit",
                                    "params": {
                                        "in_features": 4096,
                                        "out_features": 1024,
                                        "bias": False
                                    }
                                },
                                "v_proj": {
                                    "type": "Linear4bit",
                                    "params": {
                                        "in_features": 4096,
                                        "out_features": 1024,
                                        "bias": False
                                    }
                                },
                                "o_proj": {
                                    "type": "Linear4bit",
                                    "params": {
                                        "in_features": 4096,
                                        "out_features": 4096,
                                        "bias": False
                                    }
                                },
                                "rotary_emb": {
                                    "type": "LlamaRotaryEmbedding"
                                }
                            },
                            "mlp": {
                                "LlamaMLP": {
                                    "gate_proj": {
                                        "type": "Linear4bit",
                                        "params": {
                                            "in_features": 4096,
                                            "out_features": 14336,
                                            "bias": False
                                        }
                                    },
                                    "up_proj": {
                                        "type": "Linear4bit",
                                        "params": {
                                            "in_features": 4096,
                                            "out_features": 14336,
                                            "bias": False
                                        }
                                    },
                                    "down_proj": {
                                        "type": "Linear4bit",
                                        "params": {
                                            "in_features": 14336,
                                            "out_features": 4096,
                                            "bias": False
                                        }
                                    },
                                    "act_fn": "SiLU"
                                }
                            },
                            "input_layernorm": {
                                "type": "LlamaRMSNorm",
                                "params": {
                                    "shape": (4096,),
                                    "eps": 1e-05
                                }
                            },
                            "post_attention_layernorm": {
                                "type": "LlamaRMSNorm",
                                "params": {
                                    "shape": (4096,),
                                    "eps": 1e-05
                                }
                            }
                        }
                    }
                } for _ in range(32)  # Create 32 layers
            ],
            "norm": {
                "type": "LlamaRMSNorm",
                "params": {
                    "shape": (4096,),
                    "eps": 1e-05
                }
            },
            "rotary_emb": {
                "type": "LlamaRotaryEmbedding"
            }
        },
        "lm_head": {
            "type": "Linear",
            "params": {
                "in_features": 4096,
                "out_features": 128256,
                "bias": False
            }
        }
    }
}

lora_r = 16
lora_alpha = 0.04390
lora_dropout = False

peft_config_dict = {
    "lora_alpha": lora_alpha,
    "lora_dropout": lora_dropout,
    "r": lora_r,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
}


# Load the model and data preprocessing components
with open('C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//src//server_src//Diabetes_model.pkl', 'rb') as file:
    d_model = pickle.load(file)

# Load and prepare the dataset for label encoding and scaling
df = pd.read_csv("C:/Users/hp/Desktop/MyArchive/Code/Virtual_Diabetalogist/dataset/diabetes_prediction/diabetes_prediction_dataset.csv")
label_encoder_gender = LabelEncoder()
label_encoder_smoking = LabelEncoder()

# Encode the training data
df['gender'] = label_encoder_gender.fit_transform(df['gender'])
df['smoking_history'] = label_encoder_smoking.fit_transform(df['smoking_history'])

# Prepare the StandardScaler and PCA
scaler = StandardScaler()
pca = PCA()

# Fit the scaler and PCA
X_train = df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
scaler.fit(X_train)
X_scaled = scaler.transform(X_train)
pca.fit(X_scaled)

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    
    # Prepare the input DataFrame
    custom_df = pd.DataFrame([data])
    
    # Encode the 'gender' and 'smoking_history' columns
    custom_df['gender'] = label_encoder_gender.transform([custom_df['gender'][0]])
    custom_df['smoking_history'] = label_encoder_smoking.transform([custom_df['smoking_history'][0]])

    # Standardize the custom data
    custom_X = scaler.transform(custom_df[['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']])
    
    # Apply PCA transformation
    custom_X_pca = pca.transform(custom_X)

    # Make predictions using the trained model
    custom_predictions = d_model.predict(custom_X_pca)

    # Prepare the response
    response = {
        'predictions': [
            'not predicted to have diabetes' if pred == 0 else 'predicted to have diabetes' for pred in custom_predictions
        ]
    }
    
    return jsonify(response)

class PDF(FPDF):
    def header(self):
            # Set the position for the logo
            self.set_y(10)  # Set y position
            self.set_x(10)  # Set x position

            # Add company logo (with a reduced size)
            self.image('C:/Users/hp/Desktop/MyArchive/Code/Virtual_Diabetalogist/public/logo.png', x=10, y=8, w=14)  # Adjust size as needed

            # Move to the right to position the heading next to the logo
            self.set_x(30)  # Adjust x position based on the logo's width

            # Add the heading next to the logo
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'Virtual Diabetologist', ln=False, align='L')  # 'ln=False' keeps it on the same line

            # Add margin below the header (move the cursor to the next line)
            self.ln(20)  # Adjust to add space after the header (increase value for more margin)

    def footer(self):
        self.set_y(-15)  # Position at 1.5 cm from bottom
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')


# Endpoint to handle chat history summarization and generate PDF
@app.route('/chat-summary', methods=['POST'])
def chat_summary():
    data = request.get_json()
    chat_history = data.get('chat_history')

    if not chat_history:
        return jsonify({'error': 'No chat history provided'}), 400

    # Summarize the chat history using the LLM
    try:
        # Format the prompt for the LLM to summarize the chat history
        summary_prompt = f"Summarize the following chat history into a concise medical report for diabetes patients and at the top give the heading of 'AI DIAGNOSIS REPORT':\n\n{chat_history}"
        llm_response = llm.invoke(summary_prompt)
        summary_text = llm_response.content  # LLM generated summary
    except Exception as e:
        print(f"Error during LLM invocation: {e}")
        return jsonify({'error': 'Error summarizing chat history'}), 500

    # Generate a PDF from the summary
    pdf = PDF()
    pdf.add_page()


    # Process summary_text to identify and format headings
    lines = summary_text.split('\n')
    patient_data = []

    for line in lines:
        # Check for bold headings
        if line.startswith("**") and line.endswith("**"):
            heading = line[2:-2]  # Extract heading text without the **
            pdf.set_font('Arial', 'B', 16)  # Set font for headings
            pdf.cell(0, 10, heading, ln=True)  # Add heading to PDF
            pdf.set_font('Arial', '', 12)  # Reset to normal font
        # Check for patient data marked with a single star
        elif line.startswith('*') and line.endswith('*'):
            content = line[1:-1].strip()  # Extract content without the *
            patient_data.append(content)  # Add the content to patient_data list
        else:
            pdf.set_font('Arial', '', 12)  # Ensure normal font for regular text
            pdf.multi_cell(0, 10, line)  # Add normal text to PDF

    # Prepare patient information in pairs
    if patient_data:
        # Assuming the patient data is in pairs, e.g., ["Name", "John Doe", "Age", "45"]
        patient_info_pairs = [(patient_data[i], patient_data[i + 1]) for i in range(0, len(patient_data), 2)]

        # Add a table for patient information (sample structure)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Patient Information', ln=True)  # Section title
        pdf.set_font('Arial', '', 12)

        # Draw the table
        pdf.set_fill_color(200, 220, 255)  # Light blue fill color
        for key, value in patient_info_pairs:
            pdf.cell(90, 10, key, border=1, fill=True)  # Key (e.g., "Name")
            pdf.cell(90, 10, value, border=1, fill=True)  # Value (e.g., "John Doe")
            pdf.ln()  # Move to next line

    # Save the PDF to a BytesIO stream
    pdf_output = io.BytesIO()
    pdf_bytes = pdf.output(dest='S').encode('latin1')  # Encode to bytes
    pdf_output.write(pdf_bytes)  # Write PDF bytes to the BytesIO stream
    pdf_output.seek(0)  # Move the pointer to the start of the stream

    # Manually create a response and set the headers
    response = make_response(pdf_output.read())
    response.headers.set('Content-Type', 'application/pdf')
    response.headers.set('Content-Disposition', 'attachment', filename='chat_summary.pdf')

    return response


def vector_embedding():
    persist_directory = "C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//src//server_src//chroma_db"

    if os.path.exists(persist_directory):
        vectors = Chroma(collection_name="local-rag", persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"))
        print("Vectors loaded from DB")
        return vectors

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    
    file_path = "C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//dataset//merged.pdf"
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
    user_input = data.get('prompt')
    
    if user_input:
        # Update conversation history
        history.append(("human", user_input))
        if len(history) > 12:
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
        return response.get('answer', '')  
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ''
    
'''This function can be useful in determining the output size of a convolutional layer,
given the input dimensions and the convolutional layer's parameters.'''

def findConv2dOutShape(hin,win,conv,pool=2):
    kernel_size = conv.kernel_size
    stride=conv.stride
    padding=conv.padding
    dilation=conv.dilation

    hout=np.floor((hin+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    wout=np.floor((win+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)

    if pool:
        hout/=pool
        wout/=pool
    return int(hout),int(wout)

# Define Architecture For Retinopathy Model
class CNN_Retino(nn.Module):
    
    def __init__(self, params):
        
        super(CNN_Retino, self).__init__()
    
        Cin,Hin,Win = params["shape_in"]
        init_f = params["initial_filters"] 
        num_fc1 = params["num_fc1"]  
        num_classes = params["num_classes"] 
        self.dropout_rate = params["dropout_rate"] 
        
        # CNN Layers
        self.conv1 = nn.Conv2d(Cin, init_f, kernel_size=3)
        h,w=findConv2dOutShape(Hin,Win,self.conv1)
        self.conv2 = nn.Conv2d(init_f, 2*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv2)
        self.conv3 = nn.Conv2d(2*init_f, 4*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv3)
        self.conv4 = nn.Conv2d(4*init_f, 8*init_f, kernel_size=3)
        h,w=findConv2dOutShape(h,w,self.conv4)
        
        # compute the flatten size
        self.num_flatten=h*w*8*init_f
        self.fc1 = nn.Linear(self.num_flatten, num_fc1)
        self.fc2 = nn.Linear(num_fc1, num_classes)

    def forward(self, X):
        X = F.relu(self.conv1(X)) 
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, self.num_flatten)
        X = F.relu(self.fc1(X))
        X = F.dropout(X, self.dropout_rate)
        X = self.fc2(X)
        return F.log_softmax(X, dim=1)  # Apply log_softmax here




# Define the transformation (same as used in training)
transform = transforms.Compose(
    [
        transforms.Resize((255, 255)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

# Load the pretrained model
model = torch.load("C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//src//server_src//Retino_model.pt")
model.eval()  # Set the model to evaluation mode

# Move the model to the GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the pretrained model
model = torch.load("C://Users//hp//Desktop//MyArchive//Code//Virtual_Diabetalogist//src//server_src//Retino_model.pt")
model.eval()  # Set the model to evaluation mode

# Move the model to the GPU device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def load_image_from_base64(base64_str):
    """Convert base64 string to a PIL Image and apply transformations."""
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data)).convert("RGB")
    image = transform(image)  # Apply the transformations
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image_tensor):
    """Predict the class (retinopathy or not) from the image tensor."""
    image_tensor = image_tensor.to(device)  # Move to GPU/CPU
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        return predicted_class

# Define the detect route
@app.route('/detect', methods=['POST'])
def detect():
    """Receive an image from the NodeJS server and return prediction."""
    data = request.get_json()
    
    # Check if 'image' is in the request
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400

    # Decode the base64 image and load it into a tensor
    image_base64 = data['image']
    try:
        image_tensor = load_image_from_base64(image_base64)
    except Exception as e:
        return jsonify({"error": f"Invalid image format: {str(e)}"}), 400

    # Predict if the image shows diabetic retinopathy
    predicted_class = predict(image_tensor)
    
    # Create a response based on the prediction
    if predicted_class == 0:
        result = "You have diabetic retinopathy."
    else:
        result = "You don't have diabetic retinopathy."

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)