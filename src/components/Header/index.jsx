import React from 'react';
import { Link } from 'react-router-dom';
import { UserButton, useUser } from '@clerk/clerk-react';
import './Header.css';  // Add this for styling

function Header() {
    const { user, isSignedIn } = useUser();

    return (
        <div className="transparent-header">
            <div className="logo-container">
                <img src="./logo.svg" alt="Logo" />
                <div className="logo-glow"></div> 
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
