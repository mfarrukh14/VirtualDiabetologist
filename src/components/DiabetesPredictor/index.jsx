import React, { useState } from 'react';
import axios from 'axios';
import Header from '../Header';
import Footer from '../Footer';
import visionLogo from '../../../public/retina.jpg';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faSearch, faTimes } from '@fortawesome/free-solid-svg-icons';
import loadingGif from '../../../public/mesh3.gif'; // Import your loading GIF


const DiabetesPrediction = () => {
  const [formData, setFormData] = useState({
    gender: '',
    age: 10,
    hypertension: '',
    heart_disease: '',
    smoking_history: '',
    bmi: 14.0,
    HbA1c_level: 3.0,
    blood_glucose_level: 50,
  });
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showGif, setShowGif] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
    setResult(null); // Clear result on input change
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setLoading(true);
      setShowGif(true); // Show the GIF immediately

      const response = await axios.post('http://localhost:3000/predict', {
        ...formData,
        heart_disease: formData.heart_disease === 'Yes' ? 1 : 0,
        hypertension: formData.hypertension === 'Yes' ? 1 : 0, // Convert hypertension to 0 or 1
      });

      // Simulate a delay for the loading GIF
      setTimeout(() => {
        setResult(response.data.predictions[0]);
        setShowGif(false); // Hide the GIF after processing
      }, 3000);
    } catch (error) {
      console.error('Error fetching prediction:', error);
      alert('Failed to get the prediction. Please try again.');
      setShowGif(false); // Hide the GIF in case of error
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setResult(null);
    setShowGif(false);
  };

  return (
    <>
      <Header isInverted={true} />
      <div className="flex flex-col items-start justify-center min-h-screen bg-black text-white pt-24 pb-16">
        {showGif ? (
          <div className="flex flex-col items-center justify-center w-full">
            <img src={loadingGif} alt="Loading..." className="w-96 h-96" />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center w-full mt-3 mb-5">
            <h1 className="text-3xl font-bold text-slate-200 mb-6 p-5">Diabetes Prediction</h1>
            <form onSubmit={handleSubmit} className="rounded-lg p-6 w-96 flex flex-col bg-black shadow-[10px_10px_70px_rgba(255,255,255,0.2)] border-t-2 border-l-2 border-gray-950">
              {/* Row for Gender and Age */}
              <div className="flex w-full mb-4">
                <div className="flex flex-col w-full p-1">
                  <label htmlFor="gender" className="mb-1 text-slate-200 font-semibold">Gender</label>
                  <select
                    id="gender"
                    name="gender"
                    value={formData.gender}
                    onChange={handleChange}
                    className="p-2 border border-gray-700 rounded bg-black"
                    required
                  >
                    <option value="">Select Gender</option>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>
                <div className="flex flex-col w-1/2 p-1">
                  <label htmlFor="age" className="mb-1 text-slate-200 font-semibold">Age</label>
                  <input
                    id="age"
                    name="age"
                    type="number"
                    value={formData.age}
                    onChange={handleChange}
                    min="10"
                    max="150"
                    className="p-2 border border-gray-700 rounded bg-black"
                    required
                    step="1"
                  />
                </div>
              </div>

              {/* Row for Hypertension and Heart Disease */}
              <div className="flex w-full mb-4">
                <div className="flex flex-col w-1/2 p-1">
                  <label htmlFor="hypertension" className="mb-1 text-slate-200 font-semibold">Hypertension</label>
                  <select
                    id="hypertension"
                    name="hypertension"
                    value={formData.hypertension}
                    onChange={handleChange}
                    className="p-2 border border-gray-700 rounded bg-black"
                    required
                  >
                    <option value="">Select Hypertension</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
                <div className="flex flex-col w-1/2 p-1">
                  <label htmlFor="heart_disease" className="mb-1 text-slate-200 font-semibold">Heart Disease</label>
                  <select
                    id="heart_disease"
                    name="heart_disease"
                    value={formData.heart_disease}
                    onChange={handleChange}
                    className="p-2 border border-gray-700 rounded bg-black"
                    required
                  >
                    <option value="">Select Heart Disease</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
              </div>

              {/* Row for Smoking History and BMI */}
              <div className="flex w-full mb-4">
                <div className="flex flex-col w-full p-1">
                  <label htmlFor="smoking_history" className="mb-1 text-slate-200 font-semibold">Smoking History</label>
                  <select
                    id="smoking_history"
                    name="smoking_history"
                    value={formData.smoking_history}
                    onChange={handleChange}
                    className="p-2 border border-gray-700 rounded bg-black"
                    required
                  >
                    <option value="">Select Smoking History</option>
                    <option value="never">Never</option>
                    <option value="current">Current</option>
                    <option value="not current">Not Current</option>
                    <option value="former">Former</option>
                    <option value="No Info">No Info</option>
                  </select>
                </div>
                <div className="flex flex-col w-1/2 p-1">
                  <label htmlFor="bmi" className="mb-1 text-slate-200 font-semibold">BMI</label>
                  <input
                    id="bmi"
                    name="bmi"
                    type="number"
                    value={formData.bmi}
                    onChange={handleChange}
                    step="0.5"
                    min="14.0"
                    max="45.0"
                    className="p-2 border border-gray-700 rounded bg-black"
                    required
                  />
                </div>
              </div>

              {/* Row for HbA1c Level and Blood Glucose Level */}
              <div className="flex w-full mb-4">
                <div className="flex flex-col w-1/2 p-1">
                  <label htmlFor="HbA1c_level" className="mb-1 text-slate-200 font-semibold">HbA1c Level</label>
                  <input
                    id="HbA1c_level"
                    name="HbA1c_level"
                    type="number"
                    value={formData.HbA1c_level}
                    onChange={handleChange}
                    step="0.5"
                    min="3.0"
                    max="16.0"
                    className="p-2 border border-gray-400 rounded bg-black"
                    required
                  />
                </div>
                <div className="flex flex-col w-1/2 p-1">
                  <label htmlFor="blood_glucose_level" className="mb-1 text-slate-200 font-semibold">Blood Glucose Level</label>
                  <input
                    id="blood_glucose_level"
                    name="blood_glucose_level"
                    type="number"
                    value={formData.blood_glucose_level}
                    onChange={handleChange}
                    step="1"
                    min="50"
                    max="400"
                    className="p-2 border border-gray-400 rounded bg-black"
                    required
                  />
                </div>
              </div>

              {/* Submit and Clear Buttons */}
              <div className="flex justify-center">
                <button
                  type="submit"
                  className="bg-gradient-to-tr w-1/2 font-bold text-xl from-sky-400 to-violet-700 text-white p-2 rounded hover:scale-125 hover:from-blue-500 hover:to-purple-500 transition duration-300 ease-in-out"
                >
                  Predict
                </button>
              </div>
            </form>


            {result !== null && (
              <div className="mt-6 p-4 bg-white shadow-md rounded-lg relative">
                <button
                  onClick={handleClear}
                  className="absolute top-2 right-2 text-gray-500 hover:text-red-600 transition duration-300"
                >
                  <FontAwesomeIcon icon={faTimes} />
                </button>
                <h2 className="text-lg font-semibold text-gray-800">Predicted Result</h2>
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

export default DiabetesPrediction;
