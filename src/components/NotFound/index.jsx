import React from 'react';
import { Link } from 'react-router-dom';
import Header from '../Header';
import backgroundImage from '../../../public/404-background.jpg'; // Import the image

const NotFound = () => {
  return (
    <div 
      className="h-screen w-screen bg-cover bg-center flex flex-col items-center justify-center text-white relative"
      style={{ backgroundImage: `url(${backgroundImage})` }} // Use imported image
    >
      <Header isInverted={true} />
      <div className="absolute inset-0 bg-black opacity-60"></div>
      <div className="relative z-10 flex flex-col items-center">
        <h1 className="text-6xl font-bold mb-4">404 - Page Not Found</h1>
        <p className="text-2xl mb-8">Sorry, the page you're looking for doesn't exist.</p>
        <Link to="/">
          <button className="px-6 py-3 bg-gradient-to-r from-gray-800 to-gray-600 hover:from-gray-700 hover:to-gray-500 text-white text-lg rounded-md transition duration-300">
            Return to Home
          </button>
        </Link>
      </div>
    </div>
  );
};

export default NotFound;
