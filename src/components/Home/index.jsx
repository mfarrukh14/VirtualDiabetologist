import React, { useState, useEffect } from 'react';
import Footer from '../Footer';
import Header from '../Header/index';
import { Link } from 'react-router-dom';
import { useUser } from '@clerk/clerk-react';

function Home() {
  const { user, isSignedIn } = useUser();

  const videos = ['/videos/v2.mp4', '/videos/v3.mp4', '/videos/v4.mp4', '/videos/v5.mp4'];
  const quotes = [
    "Diabetes is not a choice, but we can choose to fight it.",
    "Managing diabetes is hard, but never harder than giving up.",
    "With every small step, you’re taking control of your diabetes.",
    "Diabetes doesn’t define you, your resilience does.",
    "Living with diabetes is a reminder to value every healthy moment.",
  ];

  const [currentVideoIndex, setCurrentVideoIndex] = useState(0);
  const [currentQuoteIndex, setCurrentQuoteIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);
  const [blinkingCursor, setBlinkingCursor] = useState(true);

  const typingSpeed = 100;
  const erasingSpeed = 50;
  const delayBetweenQuotes = 5000;

  const nextVideo = () => {
    setCurrentVideoIndex((prevIndex) => (prevIndex + 1) % videos.length);
  };

  useEffect(() => {
    if (isTyping) {
      if (displayedText.length < quotes[currentQuoteIndex].length) {
        const timeout = setTimeout(() => {
          setDisplayedText((prevText) => prevText + quotes[currentQuoteIndex][prevText.length]);
        }, typingSpeed);
        return () => clearTimeout(timeout);
      } else {
        const eraseTimeout = setTimeout(() => {
          setIsTyping(false);
        }, delayBetweenQuotes);
        return () => clearTimeout(eraseTimeout);
      }
    } else {
      if (displayedText.length > 0) {
        const timeout = setTimeout(() => {
          setDisplayedText((prevText) => prevText.slice(0, -1));
        }, erasingSpeed);
        return () => clearTimeout(timeout);
      } else {
        setCurrentQuoteIndex((prevIndex) => (prevIndex + 1) % quotes.length);
        setIsTyping(true);
      }
    }
  }, [displayedText, isTyping, currentQuoteIndex]);

  useEffect(() => {
    const cursorInterval = setInterval(() => {
      setBlinkingCursor((prev) => !prev);
    }, 500);

    return () => clearInterval(cursorInterval);
  }, []);

  useEffect(() => {
    const videoElement = document.getElementById('background-video');
    videoElement.addEventListener('ended', nextVideo);

    return () => {
      videoElement.removeEventListener('ended', nextVideo);
    };
  }, []);

  return (
    <div className="relative w-full h-screen overflow-hidden">
      <div className="relative w-full h-full">
        <video
          id="background-video"
          autoPlay
          loop={false}
          muted
          playsInline
          preload="auto"
          className="absolute w-full h-full object-cover"
          src={videos[currentVideoIndex]}
        />
        <div className="absolute inset-0 bg-black bg-opacity-50"></div>
        <div className="relative z-10 flex flex-col items-center justify-center h-full text-center text-white">
          <h1 className="quote text-5xl md:text-4xl sm:text-3xl xs:text-2xl leading-tight mt-32 mb-12">
            {displayedText}
            <span className="cursor">{blinkingCursor ? '_' : ' '}</span>
          </h1>
          <Link to={isSignedIn ? '/Chatbot' : '/SignInPage'}>
            <button className="mt-16 bg-gradient-to-r from-purple-500 to-blue-500 text-white font-semibold text-lg py-3 px-6 rounded-lg shadow-lg hover:from-blue-500 hover:to-purple-500 transition duration-300 ease-in-out">
              Talk with Our ChatBot
            </button>
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Home;
