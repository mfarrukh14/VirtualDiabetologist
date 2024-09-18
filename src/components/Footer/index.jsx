import React from 'react';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="footer">
    <div className="footer-content">
      <ul className="footer-links">
        <li><a href="#about">About Us</a></li>
        <li><a href="#services">Services</a></li>
        <li><a href="#contact">Contact</a></li>
        <li><a href="#privacy">Privacy Policy</a></li>
      </ul>
      <p>&copy; 2024 Farrukh & Anas. All rights reserved.</p>
    </div>
  </footer>
  )
}
