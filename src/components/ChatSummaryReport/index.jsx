import React, { useState } from 'react';
import { useUser } from '@clerk/clerk-react';
import Header from '../Header';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFileDownload } from '@fortawesome/free-solid-svg-icons';

const ChatSummary = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const { user, isLoaded } = useUser();

    const handleDownloadSummary = async () => {
        setLoading(true);
        setError('');
    
        try {
            const response = await fetch(`http://localhost:3000/chat-summary/${user.id}`, {
                method: 'GET',
            });
    
            // Use a timeout to add a 2-second delay before proceeding
            await new Promise(resolve => setTimeout(resolve, 2000));
    
            if (!response.ok) {
                const errorData = await response.json();
                // Check if the error message is due to insufficient chat history
                if (response.status === 400 && errorData.message === 'Chat history is not long enough to form a conclusive report') {
                    setError('Chat history is not long enough to form a conclusive report.');
                } else {
                    throw new Error('Failed to fetch summary.');
                }
                return; // Exit the function if there's an error
            }
    
            const blob = await response.blob();
            const url = window.URL.createObjectURL(new Blob([blob]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `${user.firstName}_AI_Diagnosis.pdf`);
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
        } catch (err) {
            console.error('Error downloading summary:', err);
            // Use a timeout to add a 2-second delay before showing error message
            await new Promise(resolve => setTimeout(resolve, 2000));
            setError('Error downloading summary. Please try again later.');
        } finally {
            setLoading(false);
        }
    };
    

    return (
        <>
            <Header isInverted={true} />
            {isLoaded ? (
                <div className="min-h-screen flex items-center justify-center bg-black py-12 px-4 sm:px-6 lg:px-8">
                    {loading ? (
                        <div className="flex flex-col items-center">
                            <img src="./reportLoading.gif" alt="Loading..." />
                        </div>
                    ) : (
                        <div className="max-w-80 w-full space-y-8">
                            <div className="relative p-6 bg-white rounded-lg shadow-[6px_6px_50px_rgba(200,190,255,0.8)]">
                                <h2 className="text-center text-2xl font-extrabold text-gray-900">
                                    Download AI Diagnosis Report
                                </h2>
                                <p className="mt-2 text-center text-sm text-gray-600">
                                    Download report for {user.firstName}
                                </p>

                                {error && <p className="text-red-600 text-sm mt-4">{error}</p>}

                                <div className="mt-8 flex justify-center">
                                    <button
                                        type="button"
                                        onClick={handleDownloadSummary}
                                        className="group relative w-48 flex flex-row font-bold justify-center items-center py-2 border border-transparent text-base rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
                                        disabled={loading}
                                    >
                                        <FontAwesomeIcon icon={faFileDownload} className="mr-2" />
                                        {loading ? 'Processing...' : 'Download Report'}
                                    </button>
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            ) : (
                <div className='bg-black w-full min-h-full flex justify-center items-center'>
                    <h3 className='text-white'>Loading the user</h3>
                </div>
            )}
        </>
    );
};

export default ChatSummary;
