import torch
from PIL import Image
import seaborn as sns
import numpy as np
sns.set(style='darkgrid')
import torchvision.transforms as transforms  
import torch.nn as nn 
import torch.nn.functional as F
import warnings
import base64
from io import BytesIO
from flask import Flask, request, jsonify
warnings.filterwarnings('ignore')


# Initialize the Flask app
app = Flask(__name__)


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

# Start the Flask app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)