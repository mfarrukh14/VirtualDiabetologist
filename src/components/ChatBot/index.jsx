import React, { useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import Header from '../Header/index';
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
            toggleModal();
        }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setIsLoading(true);

        const res = await fetch('http://127.0.0.1:5000/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt })
        });

        const data = await res.json();
        setIsLoading(false);

        if (res.ok) {
            setChatHistory((prevHistory) => [...prevHistory, { prompt, response: data.response }]);
            setPrompt('');
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
            <div className="font-sans text-white bg-gray-900 h-screen w-screen flex flex-col justify-between pt-16 mb-10"> {/* Changed pt-36 to pt-16 */}
                <div className="overflow-y-auto flex-1 px-5 pb-16 chat-container"> {/* Added class chat-container */}
                    <div className="text-center mb-5 mt-60">
                        <h1 className="hlo-txt text-5xl font-thin bg-gradient-to-r from-blue-600 via-purple-500 to-red-400 inline-block text-transparent bg-clip-text">
                            Hello, {user.firstName}
                        </h1>
                        <h1 className="afr-txt text-5xl font-thin bg-clip-text font-sans mt-4">
                            How can I help you today?
                        </h1>
                    </div>
                    <div className="chat-history w-full text-left"> {/* Full width chat history */}
                        {chatHistory.map((chat, index) => (
                            <div key={index} className="chat-item my-5 p-4 bg-white bg-opacity-10 text-gray-300 rounded-lg">
                                <p><i><b>{chat.prompt}</b></i><br />---------------------------</p>
                                <p>{chat.response}</p>
                            </div>
                        ))}
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
                            className="dark-input w-full p-3 rounded-lg bg-gray-800 border border-gray-600 text-white pr-12"
                            autoComplete="off"
                        />
                        <button type="button" onClick={toggleModal} className="form-btn absolute right-12 top-1/2 transform -translate-y-1/2 text-xl">
                            📜
                        </button>
                        <button type="submit" className="arrow-btn absolute right-4 top-1/2 transform -translate-y-1/2 text-xl" disabled={isLoading}>
                            {isLoading ? (
                                <div className="w-5 h-5 border-2 border-t-transparent border-purple-400 rounded-full animate-spin"></div>
                            ) : (
                                '➤'
                            )}
                        </button>
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
                                {['bloodSugar', 'heartRate', 'age', 'glucoseLevels', 'hb1ac'].map((field) => (
                                    <div key={field} className="form-group mb-3">
                                        <label className="text-sm text-gray-300">{field.replace(/([A-Z])/g, ' $1')}:</label>
                                        <input
                                            type="text"
                                            name={field}
                                            value={formData[field]}
                                            onChange={handleFormChange}
                                            className="w-full p-2 rounded-md bg-gray-700 border border-gray-600 text-white"
                                            autoComplete="off"
                                        />
                                    </div>
                                ))}
                                <button type="button" onClick={handleFormSubmit} className="submit-btn w-full py-2 bg-blue-500 text-white rounded-md mt-4 hover:bg-blue-700">
                                    Submit
                                </button>
                            </form>
                        </div>
                    </div>
                )}
            </div>
        </>
    );
}
