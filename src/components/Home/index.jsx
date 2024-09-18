import React, { useState, useEffect } from 'react';
import Footer from '../Footer';
import Header from '../Header/index';
import { Link } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';
import './Home.css';

function Home() {
  const { user, isSignedIn } = useUser();

  // Array of video sources
  const videos = [
    '/videos/v2.mp4',
    '/videos/v3.mp4',
    '/videos/v4.mp4',
  ];

  // State to hold the current video index
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);

  // Function to go to the next video in the array
  const nextVideo = () => {
    setCurrentVideoIndex((prevIndex) => (prevIndex + 1) % videos.length);
  };

  useEffect(() => {
    const videoElement = document.getElementById('background-video');

    // Event listener to detect when the video ends and move to the next one
    videoElement.addEventListener('ended', nextVideo);

    // Cleanup listener when component unmounts
    return () => {
      videoElement.removeEventListener('ended', nextVideo);
    };
  }, []);

  return (
    <div className="home-container">
      <div className="video-overlay">
        <video
          id="background-video"
          autoPlay
          loop={false} 
          muted
          playsInline
          preload="auto"
          className="background-video"
          src={videos[currentVideoIndex]} 
        />
        <div className="overlay"></div>
        <div className="content">
          <h1 className="quote">Diabetes is not a choice, but we can choose to fight it.</h1>
          <Link to={isSignedIn ? '/Chatbot' : '/SignInPage'}>
            <button className="gradient-button">Talk with Our ChatBot</button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Home;
