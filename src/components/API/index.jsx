import React, { useEffect, useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import axios from 'axios';

const API = () => {
    const { user } = useUser(); 
    const [apiKeys, setApiKeys] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const [modalMessage, setModalMessage] = useState('');

    useEffect(() => {
        if (user) { // Check if user is available
            fetchApiKeys();
        }
    }, [user]);

    const fetchApiKeys = async () => {
        setLoading(true);
        try {
            const userIdString = String(user.id);
            const response = await axios.get(`http://localhost:3000/api-keys`, {
                headers: {
                    'user-id': userIdString,
                },
            });
            setApiKeys(response.data);
        } catch (error) {
            console.error('Error fetching API keys:', error);
        } finally {
            setLoading(false);
        }
    };

    const createApiKey = async () => {
        if (apiKeys.length >= 5) {
            setModalMessage('You can only have a maximum of 5 API keys.');
            setShowModal(true);
            return;
        }

        try {
            const userIdString = String(user.id);
            const response = await axios.post(`http://localhost:3000/api-keys`, {}, {
                headers: {
                    'user-id': userIdString,
                }
            });
            setApiKeys([...apiKeys, response.data]);
        } catch (error) {
            console.error('Error creating API key:', error);
        }
    };

    const deleteApiKey = async (keyId) => {
        try {
            const userIdString = String(user.id);
            await axios.delete(`http://localhost:3000/api-keys/${keyId}`, {
                headers: {
                    'user-id': userIdString,
                },
            });
            setApiKeys(apiKeys.filter((key) => key.id !== keyId));
        } catch (error) {
            console.error('Error deleting API key:', error);
        }
    };

    const closeModal = () => {
        setShowModal(false);
    };

    return (
        <div className="container mx-auto p-6">
            <h1 className="text-3xl font-bold mb-6">API Keys</h1>
            <button
                onClick={createApiKey}
                className="mb-4 bg-green-600 text-white py-2 px-6 rounded shadow-md hover:bg-green-700 transition duration-300"
            >
                Create New Secret Key 
            </button>
            {loading ? (
                <p className="text-gray-500">Loading...</p>
            ) : (
                <div className="grid grid-cols-1 gap-4">
                    {apiKeys.map((key) => (
                        <div key={key.id} className="flex justify-between items-center bg-gray-100 border border-gray-300 rounded-lg p-4 shadow-sm">
                            <span className="text-lg font-medium">{key.key}</span>
                            <button
                                onClick={() => deleteApiKey(key.id)}
                                className="bg-red-600 text-white py-1 px-4 rounded transition duration-300 hover:bg-red-700"
                            >
                                Delete
                            </button>
                        </div>
                    ))}
                </div>
            )}

            {/* Modal */}
            {showModal && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 backdrop-blur-sm">
                    <div className="bg-gray-800 text-white p-6 rounded-lg shadow-lg relative">
                        <h2 className="text-xl font-bold mb-4">Notification</h2>
                        <p>{modalMessage}</p>
                        <button 
                            onClick={closeModal} 
                            className="absolute top-2 right-2 text-gray-300 hover:text-white"
                        >
                            &times;
                        </button>
                    </div>
                </div>
            )}
        </div>
    );
};

export default API;
