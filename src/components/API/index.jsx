import React, { useEffect, useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import axios from 'axios';
import { FaRegCopy, FaTrash, FaPlus } from 'react-icons/fa';
import './API.css'

// Add your astronaut image import here
import astronautImage from '../../../public/astronaut.png';
import Footer from '../Footer';
import Header from '../Header';

const API = () => {
    const { user } = useUser();
    const [apiKeys, setApiKeys] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const [modalMessage, setModalMessage] = useState('');
    const [newKeyName, setNewKeyName] = useState('');
    const [copiedIndex, setCopiedIndex] = useState(null); // State to track which key was copied

    useEffect(() => {
        if (user) {
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
        // Clear the previous modal message before attempting to create a new key
        setModalMessage('');
    
        if (!newKeyName) {
            setModalMessage('Please enter a name for the API key.');
            setShowModal(true);
            return;
        }

        // Check if the new key name is unique
        const keyNameExists = apiKeys.some(key => key.name.toLowerCase() === newKeyName.toLowerCase());
        if (keyNameExists) {
            setModalMessage('This key name is already in use. Please choose a different name.');
            setShowModal(true);
            return;
        }
    
        if (apiKeys.length >= 5) {
            setModalMessage('You can only have a maximum of 5 API keys.');
            setShowModal(true);
            return;
        }
    
        try {
            const userIdString = String(user.id);
            const response = await axios.post(`http://localhost:3000/api-keys`, { name: newKeyName }, {
                headers: {
                    'user-id': userIdString,
                }
            });
            setApiKeys([...apiKeys, response.data]);
            setNewKeyName('');
            setModalMessage(''); // Clear modal message on success
            setShowModal(false);  // Close modal
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
    
            // Clear modal message if a key was successfully deleted (since it may be related to key limit)
            setModalMessage('');
            setShowModal(false);  // Close the modal if it was open
        } catch (error) {
            console.error('Error deleting API key:', error);
        }
    };
    
    const closeModal = () => {
        setModalMessage(''); // Clear the modal message when the user closes it manually
        setShowModal(false); // Close the modal
    };
    
    const formatDate = (dateString) => {
        const date = new Date(dateString);
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0'); // Months are zero-based
        const year = date.getFullYear();
        return `${day}/${month}/${year}`;
    };

    const formatSecretKey = (key) => {
        if (key.length <= 8) return key; // Return the key if it's 8 characters or less
        return `${key.slice(0, 4)}...${key.slice(-4)}`;
    };

    const handleCopy = (key, index) => {
        navigator.clipboard.writeText(key);
        setCopiedIndex(index); // Set the index of the copied key

        setTimeout(() => {
            setCopiedIndex(null); // Reset copied index after 3 seconds
        }, 3000); 
    };

    return (
        <>
        <Header isInverted={true}/>
                <div className="container min-h-screen mx-auto pt-28 px-10 py-10 bg-gray-900 text-white">
            <h1 className="text-3xl font-bold mb-6">API Keys</h1>
            <button
                onClick={() => setShowModal(true)}
                className="mb-4 flex items-center bg-green-600 text-white py-2 px-6 rounded-lg shadow-md hover:bg-green-700 transition duration-300"
            >
                <FaPlus className="mr-2" />
                Create New Secret Key
            </button>
            {loading ? (
                <p className="text-gray-500">Loading...</p>
            ) : apiKeys.length === 0 ? (
                // If no API keys, show the "AI for developers" message
                <div className="flex flex-col items-center justify-center min-h-[200px]">
                    <h2 className="text-6xl font-thin mt-16 text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-purple-500 to-red-400 mb-4">AI for developers</h2>
                    {/* Add floating astronaut */}
                    <img 
                        src={astronautImage} 
                        alt="Astronaut floating" 
                        className="w-48 h-48 animate-floating"
                    />
                </div>
            ) : (
                // Render the table if there are API keys
                <table className="min-w-full bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
                    <thead>
                        <tr>
                            <th className="py-2 px-4 border-b border-gray-700 text-center">Name</th>
                            <th className="py-2 px-4 border-b border-gray-700 text-center">Secret Key</th>
                            <th className="py-2 px-4 border-b border-gray-700 text-center">Created On</th>
                            <th className="py-2 px-4 border-b border-gray-700 text-center">Expire On</th>
                            <th className="py-2 px-4 border-b border-gray-700 text-center">Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {apiKeys.map((key, index) => (
                            <tr key={key.id} className="hover:bg-gray-700">
                                <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400">{key.name}</td>
                                <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400 relative">
                                    {copiedIndex === index ? ( // Check if this key was copied
                                        <span className="text-green-400">Copied!</span>
                                    ) : (
                                        <>
                                            <span>{formatSecretKey(key.key)}</span>
                                            <button className="ml-2 text-blue-400 hover:underline" onClick={() => handleCopy(key.key, index)}>
                                                <FaRegCopy />
                                            </button>
                                        </>
                                    )}
                                </td>
                                <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400">{formatDate(key.createdOn)}</td>
                                <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400">{formatDate(key.expireOn)}</td>
                                <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400">
                                    <button
                                        onClick={() => deleteApiKey(key.id)}
                                        className="bg-red-600 text-white py-1 px-2 rounded transition duration-300 hover:bg-red-700"
                                    >
                                        <FaTrash />
                                    </button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}

            {/* Modal for creating new API key */}
            {showModal && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-75 backdrop-blur-sm">
                    <div className="bg-gray-800 text-white p-6 w-96 rounded-lg shadow-lg relative">
                        <h2 className="text-xl font-bold mb-4">Create New API Key</h2>
                        <input 
                            type="text" 
                            placeholder="Name of the key" 
                            value={newKeyName}
                            onChange={(e) => setNewKeyName(e.target.value)}
                            className="w-full p-2 mb-4 bg-gray-700 text-white border border-gray-600 rounded"
                        />
                        <div className="flex justify-end">
                            <button 
                                onClick={closeModal} 
                                className="mr-2 bg-gray-600 text-white py-2 px-4 rounded hover:bg-gray-700 transition duration-300"
                            >
                                Cancel
                            </button>
                            <button 
                                onClick={createApiKey} 
                                className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition duration-300"
                            >
                                Create Key
                            </button>
                        </div>
                        {modalMessage && (
                            <p className="text-red-500 mt-4">{modalMessage}</p>
                        )}
                    </div>
                </div>
            )}
        </div>
        <Footer />
        </>

    );
};

export default API;
