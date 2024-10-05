import React from 'react';
import './AboutUs.css';
import Header from '../Header';

export default function AboutUs() {
    return (
        <>
            <Header isInverted={true} />
            <div className="about-container">
                <section className="about-hero">
                    <h1>About Us</h1>
                    <p>We create simple, effective solutions for a complex world.</p>
                </section>

                <section className="about-content">
                    <div className="about-card">
                        <img src="./mission.jpg" alt="Mission" className='msn-img' />
                        <h2>Our Mission</h2>
                        <p>
                            To empower individuals and businesses by providing innovative AI solutions
                            that simplify processes and drive success.
                        </p>
                    </div>

                    <div className="about-card">
                        <img src="./vision.jpg" alt="Vision" className='msn-img' />
                        <h2>Our Vision</h2>
                        <p>
                            A future where AI and healthcare go hand in hand to shape better lives for everyone. and to help healthcare professionals.
                        </p>
                    </div>

                    <div className="about-card">
                    <img src="./values.jpg" alt="Values" className='msn-img' />
                        <h2>Our Values</h2>
                        <p>
                            We believe in transparency, excellence, and a commitment to delivering high-quality solutions
                            tailored to our clients' needs.
                        </p>
                    </div>
                </section>
            </div>
        </>
    );
}
