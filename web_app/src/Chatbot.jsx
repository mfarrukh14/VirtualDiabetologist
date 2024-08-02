import React, { useState } from 'react';
import './Chatbot.css';

export default function Chatbot() {
    const [prompt, setPrompt] = useState('');
    const [response, setResponse] = useState('');

    const handlePromptChange = (e) => {
        setPrompt(e.target.value);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        
        const res = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ prompt }),
        });
        
        const data = await res.json();
        
        if (res.ok) {
            setResponse(data.response);
        } else {
            setResponse('Error: ' + data.error);
        }
    };

    return (
        <div className="App">
            <h1>Virtual Doctor</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label htmlFor="prompt">Enter your question:</label>
                    <input
                        type="text"
                        id="prompt"
                        value={prompt}
                        onChange={handlePromptChange}
                    />
                </div>
                <button type="submit">Submit</button>
            </form>
            <div className="response">
                <h2>Response:</h2>
                <p>{response}</p>
            </div>
        </div>
    );
}
