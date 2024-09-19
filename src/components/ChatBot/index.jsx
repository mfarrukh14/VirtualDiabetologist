import React, { useState } from 'react';
import './Chatbot.css';
import Header from '../Header/index'

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
        <>
            <div className='chatbot-container'>
            <Header />
                <div className="App">
                    <h1>👋🏻Welcome to the Chatroom</h1>
                    <div className="response">
                        <p>{response}</p>
                    </div>
                    <form onSubmit={handleSubmit}>
                        <div>
                            <input
                                type="text"
                                id="prompt"
                                value={prompt}
                                onChange={handlePromptChange}
                                placeholder='Enter a prompt here'
                            />
                        </div>
                        <button type="submit">Submit</button>
                    </form>
                </div>
            </div>
        </>
    );
}
