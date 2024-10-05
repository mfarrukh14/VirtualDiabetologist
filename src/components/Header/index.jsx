import React from 'react';
import { Link } from 'react-router-dom';
import { UserButton, useUser } from '@clerk/clerk-react';
import './Header.css';  // Add this for styling

function Header({isInverted}) {
    const { user, isSignedIn } = useUser();

    return (
        <div className="transparent-header">
            <div className="logo-container">
                <Link to={'/'}>
                    {isInverted ? (
                        <button className='logo-btn'>
                            <img
                                src="./logo.svg"
                                alt="Logo"
                                style={{ filter: 'invert(100%)' }}
                            />

                        </button>
                    ) : (
                        <button className='logo-btn'>
                            <img src="./logo.svg" alt="Logo" />
                        </button>
                    )}
                </Link>
                <div className="logo-glow"></div>
            </div>
            <div className='main-hdr-btn'>
                <div className='srvc-btn'>
                    <button>Our services</button>
                </div>
                <div className='abt-btn'>
                    <Link to={'/AboutUs'}>
                    <button>About Us</button>
                    </Link>
                </div>
                <div className='contact-btn'>
                    <button>Contact Us</button>
                </div>
                <div className='api-btn'>
                    <button>API</button>
                </div>
            </div>
            {isSignedIn ? (
                <div className='after-login'>
                    <Link to={'/dashboard'}>
                        <button className='hd-btn'>Dashboard</button>
                    </Link>
                    <UserButton />
                </div>
            ) : (
                <Link to={'/SignInPage'}>
                    <button className='hd-btn'>Get Started</button>
                </Link>
            )}
        </div>
    );
}

export default Header;
