import React, { useState } from 'react';
import './Chatbot.css';
import Header from '../Header/index';
import { useUser } from '@clerk/clerk-react';

export default function Chatbot() {
    const [prompt, setPrompt] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [formData, setFormData] = useState({
        bloodSugar: '',
        heartRate: '',
        age: '',
        glucoseLevels: '',
        hb1ac: ''
    });
    const [isLoading, setIsLoading] = useState(false); // New loading state
    const { isLoaded, user } = useUser();

    const toggleModal = () => setIsModalOpen(!isModalOpen);

    const handlePromptChange = (e) => {
        setPrompt(e.target.value);
    };

    const handleFormChange = (e) => {
        const { name, value } = e.target;
        setFormData((prevData) => ({ ...prevData, [name]: value }));
    };

    const handleFormSubmit = async () => {
        const res = await fetch('http://127.0.0.1:5000/update-context', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (res.ok) {
            toggleModal(); // Close the modal on successful update
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true); // Set loading to true when submitting

        const res = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });

        const data = await res.json();
        setIsLoading(false); // Set loading to false when response is received

        if (res.ok) {
            setChatHistory((prevHistory) => [...prevHistory, { prompt, response: data.response }]);
            setPrompt('');
        }
    };

    if (!isLoaded) {
        return (
            <div className="loading-container">
                <div className="loading-spinner"></div>
            </div>
        );
    }

    return (
        <div className="chatbot-container">
            <Header isInverted={true} />
            <div className="App">
                <div className='wlcm-container'>
                    <h1 className='hello-txt'>Hello, {user.firstName}</h1>
                    <h1 className='afr-txt'>How can I help you today?</h1>
                </div>
                <div className="chat-history">
                    {chatHistory.map((chat, index) => (
                        <div key={index} className="chat-item">
                            <p><i><b>{chat.prompt}</b></i><br />---------------------------</p>
                            <p>{chat.response}</p>
                        </div>
                    ))}
                </div>
            </div>
            <form onSubmit={handleSubmit} className="chat-input-form">
                <div className="input-container">
                    <input
                        type="text"
                        id="prompt"
                        value={prompt}
                        onChange={handlePromptChange}
                        placeholder='Enter a prompt here'
                        className="dark-input"
                        autoComplete="off"
                    />
                    <button type="button" className="form-btn" onClick={toggleModal}>
                        📜
                    </button>
                    <button type="submit" className="arrow-btn" disabled={isLoading}>
                        {isLoading ? (
                            <div className="loading-spinner"></div> // Loading spinner
                        ) : (
                            '➤'
                        )}
                    </button>
                </div>
            </form>

            {isModalOpen && (
                <div className="modal">
                    <div className="modal-content">
                        <span className="close-btn" onClick={toggleModal}>&times;</span>
                        <h2>Enter your vitals</h2>
                        <form>
                            <div className="form-group">
                                <label>Blood Sugar:</label>
                                <input
                                    type="text"
                                    name="bloodSugar"
                                    value={formData.bloodSugar}
                                    onChange={handleFormChange}
                                    autoComplete="off"
                                />
                            </div>
                            <div className="form-group">
                                <label>Heart Rate:</label>
                                <input
                                    type="text"
                                    name="heartRate"
                                    value={formData.heartRate}
                                    onChange={handleFormChange}
                                    autoComplete="off"
                                />
                            </div>
                            <div className="form-group">
                                <label>Age:</label>
                                <input
                                    type="text"
                                    name="age"
                                    value={formData.age}
                                    onChange={handleFormChange}
                                    autoComplete="off"
                                />
                            </div>
                            <div className="form-group">
                                <label>Glucose Levels:</label>
                                <input
                                    type="text"
                                    name="glucoseLevels"
                                    value={formData.glucoseLevels}
                                    onChange={handleFormChange}
                                    autoComplete="off"
                                />
                            </div>
                            <div className="form-group">
                                <label>Hb1Ac:</label>
                                <input
                                    type="text"
                                    name="hb1ac"
                                    value={formData.hb1ac}
                                    onChange={handleFormChange}
                                    autoComplete="off"
                                />
                            </div>
                            <button type="button" onClick={handleFormSubmit} className="submit-btn">
                                Submit
                            </button>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}
