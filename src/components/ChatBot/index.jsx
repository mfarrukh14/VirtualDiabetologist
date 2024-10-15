import React, { useState, useEffect, useRef } from 'react';
import { useUser } from '@clerk/clerk-react';
import Header from '../Header/index';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'; // Import Font Awesome Icon
import { faBroom } from '@fortawesome/free-solid-svg-icons'; // Import Broom Icon
import './Chatbot.css';

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
    const [isLoading, setIsLoading] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const { isLoaded, user } = useUser();

    const chatContainerRef = useRef(null);

    const toggleModal = () => setIsModalOpen(!isModalOpen);

    const handlePromptChange = (e) => {
        setPrompt(e.target.value);
    };

    const handleFormChange = (e) => {
        const { name, value } = e.target;
        if (/^\d*\.?\d*$/.test(value)) {
            setFormData((prevData) => ({ ...prevData, [name]: value }));
        }
    };

    const handleFormSubmit = async () => {
        const res = await fetch('http://127.0.0.1:3000/update-context', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(formData)
        });

        if (res.ok) {
            toggleModal();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsSubmitting(true);

        const res = await fetch('http://127.0.0.1:3000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'user-id': user.id },
            body: JSON.stringify({ prompt })
        });

        const data = await res.json();
        setChatHistory((prevHistory) => [...prevHistory, { prompt, response: data.response }]);

        await fetch('http://127.0.0.1:3000/chat-history', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'user-id': user.id },
            body: JSON.stringify({ prompt, response: data.response })
        });

        setPrompt('');
        setIsSubmitting(false);
    };

    useEffect(() => {
        const fetchChatHistory = async () => {
            setIsLoading(true);
            const res = await fetch(`http://127.0.0.1:3000/chat-history`, {
                method: 'GET',
                headers: { 'user-id': user.id }
            });
            const data = await res.json();
            setIsLoading(false);

            if (res.ok && data.length > 0) {
                setChatHistory(data);
            }
        };

        if (isLoaded && user && chatHistory.length === 0) {
            fetchChatHistory();
        }
    }, [isLoaded, user, chatHistory.length]);

    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [chatHistory]);

    const handleDeleteChatHistory = async () => {
        const res = await fetch('http://127.0.0.1:3000/chat-history', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json', 'user-id': user.id }
        });

        if (res.ok) {
            setChatHistory([]);
        }
    };

    if (!isLoaded) {
        return (
            <div className="flex justify-center items-center h-screen">
                <div className="loading-spinner"></div>
            </div>
        );
    }

    return (
        <>
            <Header isInverted={true} />
            <div className="font-sans text-white bg-gray-900 h-screen w-screen flex flex-col justify-between pt-16 mb-10">
                <div className="overflow-y-auto flex-1 px-5 pb-16 chat-container" ref={chatContainerRef}>
                    <div className="text-center mb-5 mt-60">
                        <h1 className="hlo-txt text-5xl font-thin bg-gradient-to-r from-blue-600 via-purple-500 to-red-400 inline-block text-transparent bg-clip-text">
                            Hello, {user.firstName}
                        </h1>
                        <h1 className="afr-txt text-5xl font-thin bg-clip-text font-sans mt-4">
                            How can I help you today?
                        </h1>
                    </div>
                    <div className="chat-history w-full text-left">
                        {isLoading ? (
                            <p className="text-center text-gray-300">Loading chat history...</p>
                        ) : (
                            chatHistory.map((chat, index) => (
                                <div key={index} className="chat-item my-3">
                                    <div className="chat-bubble-prompt flex justify-end">
                                        <div className="bubble max-w-xl -mr-20 mb-3 p-4 rounded-lg text-white bg-sky-800">
                                            <p className="font-bold">{chat.prompt}</p>
                                        </div>
                                    </div>
                                    <div className="chat-bubble-response flex justify-start">
                                        <div className="bubble max-w-2xl -ml-20 p-4 rounded-lg text-white bg-gray-700">
                                            <p>{chat.response}</p>
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                        {isSubmitting && (
                            <div className="flex justify-center mt-4">
                                <div className="spinner"></div>
                            </div>
                        )}
                    </div>
                </div>
                <form onSubmit={handleSubmit} className="fixed bottom-0 w-full bg-gray-900 py-4">
                    <div className="relative w-4/5 mx-auto">
                        <input
                            type="text"
                            id="prompt"
                            value={prompt}
                            onChange={handlePromptChange}
                            placeholder="Enter a prompt here"
                            className="dark-input w-full p-3 rounded-lg bg-gray-800 border border-gray-600 text-white pr-20 pl-12"
                            autoComplete="off"
                        />
                        <button type="button" onClick={toggleModal} className="form-btn absolute right-12 top-1/2 transform -translate-y-1/2 text-xl">
                            📜
                        </button>
                        <button
                            type="button"
                            onClick={handleDeleteChatHistory}
                            className="form-btn absolute left-3 top-1/2 transform -translate-y-1/2 text-xl hover:text-blue-500"
                        >
                            <FontAwesomeIcon icon={faBroom} />
                        </button>
                        {isSubmitting ? (
                            <div className="flex justify-center items-center absolute right-4 top-1/2 transform -translate-y-1/2 w-8 h-8">
                                <div className="w-5 h-5 border-2 border-t-transparent border-purple-400 rounded-full animate-spin"></div>
                            </div>
                        ) : (
                            <button type="submit" className="arrow-btn absolute right-4 top-1/2 transform -translate-y-1/2 text-xl">
                                ➤
                            </button>
                        )}
                    </div>
                </form>

                {isModalOpen && (
                    <div className="modal fixed inset-0 bg-black bg-opacity-50 flex justify-center items-center z-50">
                        <div className="modal-content bg-gray-800 p-6 rounded-lg animate-glow">
                            <span className="close-btn absolute top-2 right-2 text-2xl cursor-pointer" onClick={toggleModal}>
                                &times;
                            </span>
                            <h2 className="text-lg mb-4">Enter your vitals</h2>
                            <form>
                                <div className="grid grid-cols-2 gap-4 mb-4">
                                    <div className="form-group">
                                        <label className="text-sm text-gray-300">Blood Sugar (mm/dL):</label>
                                        <input
                                            type="text"
                                            name="bloodSugar"
                                            value={formData.bloodSugar}
                                            onChange={handleFormChange}
                                            placeholder="mm/dL"
                                            className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 text-white placeholder-gray-400"
                                            pattern="\d*"
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label className="text-sm text-gray-300">Heart Rate (bpm):</label>
                                        <input
                                            type="text"
                                            name="heartRate"
                                            value={formData.heartRate}
                                            onChange={handleFormChange}
                                            placeholder="bpm"
                                            className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 text-white placeholder-gray-400"
                                            pattern="\d*"
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label className="text-sm text-gray-300">Age (years):</label>
                                        <input
                                            type="text"
                                            name="age"
                                            value={formData.age}
                                            onChange={handleFormChange}
                                            placeholder="years"
                                            className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 text-white placeholder-gray-400"
                                            pattern="\d*"
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label className="text-sm text-gray-300">Glucose Levels (mg/dL):</label>
                                        <input
                                            type="text"
                                            name="glucoseLevels"
                                            value={formData.glucoseLevels}
                                            onChange={handleFormChange}
                                            placeholder="mg/dL"
                                            className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 text-white placeholder-gray-400"
                                            pattern="\d*"
                                        />
                                    </div>
                                    <div className="form-group col-span-2">
                                        <label className="text-sm text-gray-300">Hb1Ac (%):</label>
                                        <input
                                            type="text"
                                            name="hb1ac"
                                            value={formData.hb1ac}
                                            onChange={handleFormChange}
                                            placeholder="%"
                                            className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 text-white placeholder-gray-400"
                                            pattern="\d*"
                                        />
                                    </div>
                                </div>
                                <div className="flex justify-center mt-4">
                                    <button
                                        type="button"
                                        onClick={toggleModal}
                                        className="cancel-btn w-1/3 h-10 py-1.5 bg-gray-500 text-white rounded-md hover:bg-gray-600 mr-2 mt-4" // Added 'mr-2' for margin
                                    >
                                        Cancel
                                    </button>
                                    <button
                                        type="button"
                                        onClick={handleFormSubmit}
                                        className={`submit-btn w-1/3 h-10 py-1.5 text-white rounded-md mt-4 ${!Object.values(formData).some(value => value) ? 'bg-blue-500 opacity-50 cursor-not-allowed' : 'bg-blue-500 hover:bg-blue-700'}`}
                                        disabled={!Object.values(formData).some(value => value)}
                                    >
                                        Submit
                                    </button>
                                </div>
                            </form>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}
