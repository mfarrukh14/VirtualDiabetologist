import React, { useState } from 'react';
import axios from 'axios';
import Header from '../Header';
import Footer from '../Footer';
import visionLogo from '../../../public/retina.jpg';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faFileUpload, faSearch, faTimes } from '@fortawesome/free-solid-svg-icons';
import loadingGif from '../../../public/mesh2.gif'; // Import your loading GIF
import { useUser } from '@clerk/clerk-react';

const VisionHeader = () => (
  <div className="text-left mb-6 ml-4">
    <p className="text-slate-200 text-lg">A project by</p>
    <h1 className="text-6xl text-slate-200 flex items-center font-thin">
      V I S I
      <img
        src={visionLogo}
        alt="O"
        className="h-12 w-12 mx-4 mt-2 animate-pulse"
      />
      N
    </h1>
  </div>
);

const Retinopathy = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showGif, setShowGif] = useState(false);
  const {user} = useUser();

  const handleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setResult(null);
  };

  const handleDetect = async () => {
    if (!selectedFile) {
      alert('Please select an image.');
      return;
    }
    const userIdString = String(user.id);
    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      setLoading(true);
      setShowGif(true); // Show the GIF immediately

      const response = await axios.post('http://localhost:3000/upload-image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'user-id': userIdString,
      }      
      });

      // Use setTimeout to add a 2-second delay before setting the result
      setTimeout(() => {
        setResult(response.data.result);
        setShowGif(false); // Hide the GIF after 2 seconds
      }, 3000);
    } catch (error) {
      console.error('Error uploading the image:', error);
      alert('Failed to upload the image. Please try again.');
      setShowGif(false); // Hide the GIF in case of error
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setSelectedFile(null); // Clear the selected file
    setResult(null); // Clear the result
    document.getElementById('fileInput').value = ''; // Clear the file input
    setShowGif(false); // Hide the GIF when clearing
  };

  // Function to format the file name
  const formatFileName = (fileName) => {
    const extension = fileName.slice(-4); // Capture the last 4 characters for extension
    const name = fileName.slice(0, -4); // Get the name without the extension
    if (name.length > 7) {
      const formattedName = `${name.slice(0, 2)}...${name.slice(-2)}${extension}`;
      return formattedName;
    }
    return fileName;
  };

  return (
    <>
      <Header isInverted={true} />
      <div className="flex flex-col items-start justify-center min-h-screen bg-slate-950 text-white pt-24">
        <VisionHeader />
        {showGif ? (
          <div className="flex flex-col items-center justify-center w-full">
            <img src={loadingGif} alt="Loading..." className="w-96 h-96" />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center w-full mt-3 mb-5">
            <h1 className="text-3xl font-bold text-slate-200 mb-6 p-5">Diabetes Retinopathy Detection</h1>
            <div className="border-4 border-dashed border-gray-300 rounded-lg p-6 w-96 flex flex-col items-center justify-center">
              <input
                id="fileInput"
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleFileChange}
              />
              {selectedFile ? (
                <p className="mb-4 text-gray-500">{formatFileName(selectedFile.name)}</p>
              ) : (
                <p className="mb-4 text-gray-500">No file selected</p>
              )}
              <div className="flex space-x-4">
                <button
                  onClick={() => document.getElementById('fileInput').click()}
                  className="bg-blue-500 text-white font-semibold py-2 px-4 rounded hover:bg-blue-600 transition duration-300 flex items-center"
                >
                  <FontAwesomeIcon icon={faFileUpload} className="mr-2" />
                  <span className="hidden md:inline">Upload Image</span> {/* Show text only on medium and larger screens */}
                </button>
                {selectedFile && (
                  <button
                    onClick={handleDetect}
                    className="bg-green-500 text-white font-semibold py-2 px-4 rounded hover:bg-green-600 transition duration-300 flex items-center"
                    disabled={loading}
                  >
                    <FontAwesomeIcon icon={faSearch} className="mr-2" />
                    <span className="hidden md:inline">{loading ? 'Detecting...' : 'Detect'}</span> {/* Show text only on medium and larger screens */}
                  </button>
                )}
              </div>
              {loading && <p className="mt-4 text-gray-500">Processing...</p>}
            </div>
            {result && (
              <div className="mt-6 p-4 bg-white shadow-md rounded-lg relative">
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 text-gray-500 hover:text-red-600 transition duration-300"
                >
                  <FontAwesomeIcon icon={faTimes} />
                </button>
                <h2 className="text-lg font-semibold text-gray-800">Detection Result</h2>
                <p className="mt-2 text-gray-600">{result}</p>
              </div>
            )}
          </div>
        )}
      </div>
      <Footer />
    </>
  );
};

export default Retinopathy;
