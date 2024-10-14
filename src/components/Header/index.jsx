import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { UserButton, useUser } from '@clerk/clerk-react';
import './Header.css';
import { FaUser } from 'react-icons/fa';

function Header({ isInverted }) {
    const { user, isSignedIn } = useUser();
    const [isMenuOpen, setIsMenuOpen] = useState(false);

    const toggleMenu = () => {
        setIsMenuOpen(!isMenuOpen);
    };

    // Customize the UserButton appearance
    const userButtonAppearance = {
        elements: {
            userButtonAvatarBox: "w-10 h-10", // Custom width and height
            userButtonPopoverCard: "bg-blue-100", // Custom background for the popover card
            userButtonPopoverActionButton: "text-red-600", // Custom text color for action buttons
        },
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
                    &#9776; {/* Hamburger icon */}
                </button>
            </div>

            {/* Main header buttons */}
            <div className={`main-hdr-btn flex flex-col md:flex-row md:gap-8 gap-2 transition-all duration-300 ${isMenuOpen ? 'absolute right-0 top-16 bg-gray-800 rounded-md shadow-lg p-4' : 'hidden md:flex'}`}>
                <div className='srvc-btn'>
                    <Link to={'/OurServices'}>
                        <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">Our Services</button>
                    </Link>
                </div>
                <div className='abt-btn'>
                    <Link to={'/AboutUs'}>
                        <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">About Us</button>
                    </Link>
                </div>
                <div className='contact-btn'>
                    <Link to={'/ContactUs'}>
                        <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">Contact Us</button>
                    </Link>
                </div>
                <div className='api-btn'>
                    <Link to={'/API'}>
                        <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110">API</button>
                    </Link>
                </div>
            </div>

            {isSignedIn ? (
                <div className='after-login flex items-center space-x-4'>
                    <UserButton
                        appearance={userButtonAppearance} // Applying custom appearance
                        renderTrigger={({ open }) => (
                            <button onClick={open} className="flex items-center justify-center p-2 rounded-md bg-slate-700 hover:bg-slate-600">
                                <FaUser className="text-white" size={24} /> {/* User icon */}
                            </button>
                        )}
                    />
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
