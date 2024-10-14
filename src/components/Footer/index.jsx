import React from 'react';
import './Footer.css';
import { Link } from 'react-router-dom';

export default function Footer() {
  return (
    <footer className="footer">
    <div className="footer-content">
      <ul className="footer-links">
        <li><Link to={'/AboutUs'}><a>About Us</a></Link></li>
        <li><Link to={'/OurServices'}><a>Services</a></Link></li>
        <li><Link to={'/ContactUs'}><a>Contact</a></Link></li>
        <li><Link to={'/PrivacyPolicy'}><a>Privacy Policy</a></Link></li>
      </ul>
      <p>&copy; 2024 Farrukh & Anas. All rights reserved.</p>
    </div>
  </footer>
  )
}
