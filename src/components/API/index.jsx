import React, { useEffect, useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import axios from 'axios';
import { FaRegCopy, FaTrash, FaPlus } from 'react-icons/fa';
import './API.css';

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
    const [copiedIndex, setCopiedIndex] = useState(null); 
    const [keyToDelete, setKeyToDelete] = useState(null); // Track the key to delete
    const [showDeleteConfirm, setShowDeleteConfirm] = useState(false); // Track the delete confirmation modal

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
        setModalMessage('');

        if (!newKeyName) {
            setModalMessage('Please enter a name for the API key.');
            setShowModal(true);
            return;
        }

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
            setModalMessage('');
            setShowModal(false);
        } catch (error) {
            console.error('Error creating API key:', error);
        }
    };

    const confirmDeleteApiKey = (key) => {
        setKeyToDelete(key);
        setShowDeleteConfirm(true); // Show delete confirmation modal
    };

    const deleteApiKey = async () => {
        try {
            const userIdString = String(user.id);
            await axios.delete(`http://localhost:3000/api-keys/${keyToDelete.id}`, {
                headers: {
                    'user-id': userIdString,
                },
            });
            setApiKeys(apiKeys.filter((key) => key.id !== keyToDelete.id));
            setKeyToDelete(null);
            setShowDeleteConfirm(false);
        } catch (error) {
            console.error('Error deleting API key:', error);
        }
    };

    const closeModal = () => {
        setModalMessage('');
        setShowModal(false);
        setShowDeleteConfirm(false); // Close the delete confirmation modal
    };

    const formatDate = (dateString) => {
        const date = new Date(dateString);
        const day = String(date.getDate()).padStart(2, '0');
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const year = date.getFullYear();
        return `${day}/${month}/${year}`;
    };

    const formatSecretKey = (key) => {
        if (key.length <= 8) return key;
        return `${key.slice(0, 4)}...${key.slice(-4)}`;
    };

    const handleCopy = (key, index) => {
        navigator.clipboard.writeText(key);
        setCopiedIndex(index);

        setTimeout(() => {
            setCopiedIndex(null);
        }, 1000);
    };

    return (
        <>
            <Header isInverted={true} />
            <div className="container min-h-screen mx-auto pt-28 px-10 py-10 bg-slate-950 text-white">
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
                    <div className="flex flex-col items-center justify-center min-h-[200px]">
                        <h2 className="text-6xl font-thin mt-16 text-transparent bg-clip-text bg-gradient-to-r from-blue-600 via-purple-500 to-red-400 mb-4">AI for developers</h2>
                        <img
                            src={astronautImage}
                            alt="Astronaut floating"
                            className="w-48 h-48 animate-floating"
                        />
                    </div>
                ) : (
                    <table className="min-w-full bg-gray-800 border border-gray-700 rounded-lg overflow-hidden">
                        <thead>
                            <tr>
                                <th className="py-2 px-4 border-b border-gray-700 text-center">Name</th>
                                <th className="py-2 px-4 border-b border-gray-700 text-center">Secret Key</th>
                                <th className="py-2 px-4 border-b border-gray-700 text-center hidden sm:table-cell">Created On</th>
                                <th className="py-2 px-4 border-b border-gray-700 text-center hidden sm:table-cell">Expire On</th>
                                <th className="py-2 px-4 border-b border-gray-700 text-center">Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {apiKeys.map((key, index) => (
                                <tr key={key.id} className="hover:bg-gray-700">
                                    <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400">{key.name}</td>
                                    <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400 relative">
                                        {copiedIndex === index ? (
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
                                    <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400 hidden sm:table-cell">
                                        {formatDate(key.createdOn)}
                                    </td>
                                    <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400 hidden sm:table-cell">
                                        {formatDate(key.expireOn)}
                                    </td>
                                    <td className="py-2 px-4 border-b border-gray-700 text-center text-gray-400">
                                        <button
                                            onClick={() => confirmDeleteApiKey(key)} // Open delete confirmation modal
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

                {/* Create New API Key Modal */}
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

                {/* Delete Confirmation Modal */}
                {showDeleteConfirm && keyToDelete && (
                    <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-75 backdrop-blur-sm">
                        <div className="bg-gray-800 text-white p-6 w-96 rounded-lg shadow-lg">
                            <h2 className="text-xl font-bold mb-4">Warning</h2>
                            <p className="mb-4">Are you sure you want to revoke the key: "<strong>{keyToDelete.name}</strong>" ?</p>
                            <div className="flex justify-end">
                                <button
                                    onClick={closeModal}
                                    className="mr-2 bg-gray-600 text-white py-2 px-4 rounded hover:bg-gray-700 transition duration-300"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={deleteApiKey}
                                    className="bg-red-600 text-white py-2 px-4 rounded hover:bg-red-700 transition duration-300"
                                >
                                    Delete
                                </button>
                            </div>
                        </div>
                    </div>
                )}
            </div>
            <Footer />
        </>
    );
};

export default API;
