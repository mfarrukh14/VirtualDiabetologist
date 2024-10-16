import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { UserButton, useUser } from '@clerk/clerk-react';
import { FaUser, FaTimes, FaServicestack, FaInfoCircle, FaEnvelope, FaCode, FaEye } from 'react-icons/fa'; // Import the required icons
import './Header.css';
import logo from '../../../public/logo.svg'; // Import the logo image

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

    // Menu items with corresponding icons
    const menuItems = [
        { path: '/OurServices', label: 'Our Services', icon: <FaServicestack /> },
        { path: '/AboutUs', label: 'About Us', icon: <FaInfoCircle /> },
        { path: '/ContactUs', label: 'Contact Us', icon: <FaEnvelope /> },
        { path: '/API', label: 'API', icon: <FaCode /> },
        { path: '/Retinopathy', label: 'Retinopathy', icon: <FaEye /> },
    ];

    return (
        <div className="transparent-header flex items-center justify-between w-full py-4 px-6 absolute top-0 z-50">
            {/* Logo */}
            <div className="logo-container flex-shrink-0">
                <Link to={'/'}>
                    <button className='logo-btn'>
                        <img
                            src={logo} // Use the imported logo
                            alt="Logo"
                            className={isInverted ? "invert" : ""}
                        />
                    </button>
                </Link>
            </div>

            {/* Main header buttons (visible in desktop view) */}
            <div className="hidden md:flex flex-col md:flex-row md:gap-12">
                {menuItems.map(({ path, label, icon }, index) => (
                    <Link to={path} key={index}>
                        <button className="text-gray-400 hover:text-white font-bold transition-transform duration-300 hover:scale-110 flex items-center space-x-2">
                            {icon} {/* Display icon */}
                            <span>{label}</span> {/* Display button text */}
                        </button>
                    </Link>
                ))}
            </div>

            {/* User Authentication and Hamburger Button */}
            <div className='flex items-center space-x-4'>
                {isSignedIn ? (
                    <UserButton
                        appearance={userButtonAppearance}
                        renderTrigger={({ open }) => (
                            <button onClick={open} className="flex items-center justify-center p-2 rounded-md bg-slate-700 hover:bg-slate-600">
                                <FaUser className="text-white" size={24} /> {/* User icon */}
                            </button>
                        )}
                    />
                ) : (
                    <Link to={'/SignInPage'}>
                        <button className="hd-btn bg-slate-700 text-white font-bold py-2 px-4 rounded-md hover:scale-105 transition-all duration-300">
                            Get Started
                        </button>
                    </Link>
                )}

                {/* Hamburger menu button for small screens */}
                <div className="md:hidden">
                    <button onClick={toggleMenu} className="text-gray-400 focus:outline-none">
                        &#9776; {/* Hamburger icon */}
                    </button>
                </div>
            </div>

            {/* Full-screen mobile menu that appears when the hamburger icon is clicked */}
            {isMenuOpen && (
                <div className="fixed inset-0 bg-gray-800 z-40 flex flex-col"> {/* Solid background */}
                    <button
                        onClick={toggleMenu}
                        className="absolute top-4 right-4 text-white text-2xl"
                    >
                        <FaTimes /> {/* Close icon */}
                    </button>
                    <div className="flex flex-col items-center justify-center h-full space-y-12"> {/* Add space-y-* for spacing */}
                        {menuItems.map(({ path, label, icon }, index) => (
                            <Link to={path} key={index} onClick={toggleMenu}>
                                <button className="text-white text-lg font-bold flex items-center space-x-2">
                                    {icon} {/* Display icon */}
                                    <span>{label}</span> {/* Display button text */}
                                </button>
                            </Link>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default Header;