import React, { useState, useEffect } from 'react';
import './AboutUs.css';
import Header from '../Header';

export default function AboutUs() {
    // Typewriter effect state
    const fullText = "Chhanging the world through personalized digital health solutions.";
    const [displayedText, setDisplayedText] = useState("");
    const typingSpeed = 80; // Adjust typing speed here

    useEffect(() => {
        let currentIndex = 0;

        const typingInterval = setInterval(() => {
            // Check if we have reached the end of the text
            if (currentIndex < fullText.length) {
                setDisplayedText((prev) => prev + fullText.charAt(currentIndex));
                currentIndex++;
            } else {
                clearInterval(typingInterval);
            }
        }, typingSpeed);

        return () => clearInterval(typingInterval); // Cleanup on unmount
    }, [fullText]);

    return (
        <>
            <Header isInverted={true} />
            <div className="bg-black text-white min-h-screen flex flex-col justify-between pt-24">
                <div className="flex flex-col md:flex-row items-center justify-between max-w-6xl px-6 mx-auto">

                    {/* Left Section: Text */}
                    <div className="md:w-1/2 text-left md:pr-10">
                        <h1 className="text-4xl md:text-5xl font-bold mb-6">
                            <span>{displayedText}</span>
                            <span className="blinking-cursor">_</span>
                        </h1>
                        <h2 className="text-lg md:text-xl mb-6">
                            Founded by two geeks on the simple idea of creating innovative solutions that change the world, Virtual Diabetologist offers groundbreaking technology that allows users to take control of their health, making diabetes care more accessible, adaptive, and effective for everyone, everywhere.
                        </h2>
                    </div>

                    {/* Right Section: Icon */}
                    <div className="md:w-1/2 flex justify-center">
                        <img
                            src="./3dicon1.png"
                            alt="3D Icon"
                            className="w-full max-w-md h-auto"
                        />
                    </div>
                </div>

                {/* Gradient Background for Buttons */}
                <div className="w-full h-12 bg-gradient-to-r from-pink-500 to-purple-500 p-4 fixed bottom-0 left-0">
                    <div className="flex justify-center space-x-4">
                        <button className="px-4 py-2 bg-transparent text-white rounded-full text-sm md:text-md font-semibold">
                            Products and Technologies
                        </button>
                        <button className="px-4 py-2 bg-transparent text-white rounded-full text-sm md:text-md font-semibold">
                            Customer Stories
                        </button>
                        <button className="px-4 py-2 bg-transparent text-white rounded-full text-sm md:text-md font-semibold">
                            Purpose and Values
                        </button>
                        <button className="px-4 py-2 bg-transparent text-white rounded-full text-sm md:text-md font-semibold">
                            More Resources
                        </button>
                    </div>
                </div>
            </div>
        </>
    );
}
