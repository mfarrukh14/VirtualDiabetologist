import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { UserButton, useUser } from '@clerk/clerk-react';
import './Header.css'

function Header({ isInverted }) {
    const { user, isSignedIn } = useUser();
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    return (
        <div className="transparent-header flex items-center justify-between w-full py-4 px-6 absolute top-0 z-50">
            <div className="logo-container flex-shrink-0">
                <Link to={'/'}>
                    {isInverted ? (
                        <button className='logo-btn'>
                            <img
                                src="./logo.svg"
                                alt="Logo"
                                className="invert"
                            />
                        </button>
                    ) : (
                        <button className='logo-btn'>
                            <img src="./logo.svg" alt="Logo" />
                        </button>
                    )}
                </Link>
            </div>

            {/* Mobile menu button */}
            <div className="md:hidden">
                <button onClick={toggleMenu} className="text-gray-400 focus:outline-none">
                    {/* Add an icon for the menu button here (e.g., FontAwesome or a custom icon) */}
                    &#9776; {/* Hamburger icon */}
                </button>
            </div>

            {/* Main header buttons */}
            <div className={`main-hdr-btn flex flex-col md:flex-row md:gap-8 gap-2 transition-all duration-300 ${isMenuOpen ? 'absolute right-0 top-16 bg-gray-800 rounded-md shadow-lg p-4' : 'hidden md:flex'}`}>
                <div className='srvc-btn'>
                    <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">Our Services</button>
                </div>
                <div className='abt-btn'>
                    <Link to={'/AboutUs'}>
                        <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">About Us</button>
                    </Link>
                </div>
                <div className='contact-btn'>
                    <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">Contact Us</button>
                </div>
                <div className='api-btn'>
                    <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">API</button>
                </div>
            </div>

            {isSignedIn ? (
                <div className='after-login flex items-center space-x-4'>
                    <Link to={'/dashboard'}>
                        <button className="hd-btn bg-slate-700 text-white font-bold py-2 px-4 rounded-md hover:scale-105 transition-all duration-300">Dashboard</button>
                    </Link>
                    <UserButton />
                </div>
            ) : (
                <Link to={'/SignInPage'}>
                    <button className="hd-btn bg-slate-700 text-white font-bold py-2 px-4 rounded-md hover:scale-105 transition-all duration-300">Get Started</button>
                </Link>
            )}
        </div>
    );
}

export default Header;
