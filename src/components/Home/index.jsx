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
    '/videos/v5.mp4',
  ];

  // Array of quotes
  const quotes = [
    "Diabetes is not a choice, but we can choose to fight it.",
    "Managing diabetes is hard, but never harder than giving up.",
    "With every small step, you’re taking control of your diabetes.",
    "Diabetes doesn’t define you, your resilience does.",
    "Living with diabetes is a reminder to value every healthy moment."
  ];

  // State to hold the current video index
  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);

  // State to hold the current quote index
  const [currentQuoteIndex, setCurrentQuoteIndex] = useState(0);

  // State to hold the displayed text for typewriter effect
  const [displayedText, setDisplayedText] = useState('');

  // State to control whether we are typing or erasing
  const [isTyping, setIsTyping] = useState(true);

  // State to handle blinking cursor
  const [blinkingCursor, setBlinkingCursor] = useState(true);

  // Speed for typing and erasing (in milliseconds)
  const typingSpeed = 100; // Typing speed for each character
  const erasingSpeed = 50;  // Erasing speed for each character
  const delayBetweenQuotes = 5000; // Delay before switching to next quote

  // Function to go to the next video in the array
  const nextVideo = () => {
    setCurrentVideoIndex((prevIndex) => (prevIndex + 1) % videos.length);
  };

  // Function to type out the quote letter by letter
  useEffect(() => {
    if (isTyping) {
      if (displayedText.length < quotes[currentQuoteIndex].length) {
        // Continue typing next character
        const timeout = setTimeout(() => {
          setDisplayedText((prevText) => prevText + quotes[currentQuoteIndex][prevText.length]);
        }, typingSpeed);
        return () => clearTimeout(timeout);
      } else {
        // Once the full quote is typed, wait for 5 seconds and start erasing
        const eraseTimeout = setTimeout(() => {
          setIsTyping(false); // Start erasing phase
        }, delayBetweenQuotes);
        return () => clearTimeout(eraseTimeout);
      }
    } else {
      if (displayedText.length > 0) {
        // Continue erasing the characters
        const timeout = setTimeout(() => {
          setDisplayedText((prevText) => prevText.slice(0, -1));
        }, erasingSpeed);
        return () => clearTimeout(timeout);
      } else {
        // Once erasing is complete, move to the next quote and start typing again
        setCurrentQuoteIndex((prevIndex) => (prevIndex + 1) % quotes.length);
        setIsTyping(true); // Start typing the new quote
      }
    }
  }, [displayedText, isTyping, currentQuoteIndex]);

  // Handle blinking cursor
  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setBlinkingCursor((prev) => !prev); // Toggle the cursor visibility
    }, 500); // Blink every 500ms

    return () => clearInterval(cursorInterval); // Cleanup interval when component unmounts
  }, []);

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
          <h1 className="quote">
            {displayedText}
            <span className="cursor">{blinkingCursor ? '_' : ' '}</span>
          </h1>
          <Link to={isSignedIn ? '/Chatbot' : '/SignInPage'}>
            <button className="bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold text-lg py-3 px-6 rounded-lg shadow-lg hover:from-blue-500 hover:to-purple-500 transition duration-300 ease-in-out">
              Talk with Our ChatBot
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Home;
