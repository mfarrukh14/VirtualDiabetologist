# Virtual Diabetologist

Virtual Diabetologist is a healthcare web application focused on providing diabetic and retinopathy predictions, along with a chatbot that assists users based on personalized context. The app is powered by Flask (Python), Node.js, React, and several machine learning models.

## Features

- **Diabetes Prediction**: Predicts the likelihood of diabetes based on user inputs such as age, BMI, hypertension, heart disease, and more.
- **Retinopathy Prediction**: Detects diabetic retinopathy using image analysis, with support for uploading images in base64 format.
- **Chatbot**: Provides context-aware responses about diabetes and general healthcare, powered by language models and embeddings.
- **API Management**: Create and revoke APIs for integrating predictions into other applications.

## Tech Stack

### Backend
- **Flask**: Core server logic for diabetes and retinopathy predictions, as well as chatbot processing.
- **Node.js**: Middleware to manage API requests, handle file uploads, and process predictions.
- **LangChain**: Used for integrating LLMs (language models) and handling conversation flows.
- **Machine Learning Models**:
  - Diabetes prediction using a pre-trained model.
  - Image-based retinopathy detection via a CNN model.

### Frontend
- **React**: For creating the user interface.
- **Tailwind CSS**: For modern and responsive styling.
- **Font Awesome**: For icons and UI enhancements.

## Installation

### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- Node.js (for middleware and front-end)
- `pip` (for Python dependencies)
- `npm` or `yarn` (for JavaScript dependencies)

### Backend (Flask)
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/virtual-diabetologist.git
    cd virtual-diabetologist
    ```

2. Install Python dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Start the Flask server:
    ```bash
    python server.py
    ```

### Frontend (React)
1. Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```

2. Install JavaScript dependencies:
    ```bash
    npm install
    ```

3. Start the React development server:
    ```bash
    npm start
    ```

### Middleware (Node.js)
1. Navigate to the `middleware` directory:
    ```bash
    cd middleware
    ```

2. Install Node.js dependencies:
    ```bash
    npm install
    ```

3. Start the Node.js server:
    ```bash
    node server.js
    ```

## API Endpoints

### Diabetes Prediction
- **POST** `/predict`
    - Request Body:
      ```json
      {
        "gender": "Male",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 1,
        "smoking_history": "never",
        "bmi": 28.7,
        "HbA1c_level": 6.5,
        "blood_glucose_level": 120
      }
      ```
    - Response:
      ```json
      {
        "predictions": ["predicted to have diabetes"]
      }
      ```

### Retinopathy Detection
- **POST** `/detect`
    - Request Body:
      ```json
      {
        "image": "base64-encoded-image-string"
      }
      ```
    - Response:
      ```json
      {
        "result": "You have diabetic retinopathy."
      }
      ```

### Chatbot
- **POST** `/ask`
    - Request Body:
      ```json
      {
        "prompt": "What are the symptoms of diabetes?"
      }
      ```
    - Response:
      ```json
      {
        "response": "Common symptoms of diabetes include increased thirst, frequent urination, extreme hunger, etc."
      }
      ```

### Update User Context
- **POST** `/update-context`
    - Request Body:
      ```json
      {
        "bloodSugar": 180,
        "heartRate": 72,
        "age": 45,
        "glucoseLevels": 200,
        "hb1ac": 7.2
      }
      ```
    - Response:
      ```json
      {
        "message": "User data context updated successfully"
      }
      ```

## License

This project is licensed under the MIT License.

