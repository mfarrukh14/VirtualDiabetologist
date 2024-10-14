import React from 'react';
import Header from '../Header';
import Footer from '../Footer';

const PrivacyPolicy = () => {
    return (
        <>
            <Header isInverted={true} />
            <div className="bg-gray-900 text-white py-16 px-4 pt-32">
                <div className="max-w-7xl mx-auto text-center">
                    <h1 className="text-4xl font-bold mb-8 animate__animated animate__fadeIn">
                        Privacy Policy
                    </h1>
                    <p className="mb-4 text-gray-400 animate__animated animate__fadeIn">
                        Last updated: 14/10/2024
                    </p>
                    <p className="mb-6 text-gray-300 animate__animated animate__fadeIn">
                        This Privacy Policy explains how our virtual diabetologist service collects, uses, and protects your information.
                    </p>
                    
                    {/* Flex container for cards */}
                    <div className="flex flex-wrap justify-center gap-6">
                        {/* Card 1 */}
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700 w-full sm:w-80 animate__animated animate__fadeIn">
                            <h2 className="text-2xl font-semibold mb-4">Information We Collect</h2>
                            <p className="text-gray-300 mb-4">
                                We may collect the following types of information:
                            </p>
                            <ul className="list-disc list-inside text-gray-400 text-left">
                                <li>Personal information (name, email, contact number)</li>
                                <li>Health-related data (blood sugar levels, heart rate, etc.)</li>
                                <li>Usage data (how you interact with our service)</li>
                            </ul>
                        </div>

                        {/* Card 2 */}
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700 w-full sm:w-80 animate__animated animate__fadeIn">
                            <h2 className="text-2xl font-semibold mb-4">How We Use Your Information</h2>
                            <p className="text-gray-300 mb-4">
                                We use your information to:
                            </p>
                            <ul className="list-disc list-inside text-gray-400 text-left">
                                <li>Provide personalized healthcare recommendations</li>
                                <li>Improve our services and user experience</li>
                                <li>Communicate with you regarding updates and services</li>
                            </ul>
                        </div>

                        {/* Card 3 */}
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700 w-full sm:w-80 animate__animated animate__fadeIn">
                            <h2 className="text-2xl font-semibold mb-4">Data Protection</h2>
                            <p className="text-gray-300 mb-4">
                                We implement security measures to protect your personal information, including:
                            </p>
                            <ul className="list-disc list-inside text-gray-400 text-left">
                                <li>Data encryption</li>
                                <li>Access controls</li>
                                <li>Regular security audits</li>
                            </ul>
                        </div>

                        {/* Card 4 */}
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700 w-full sm:w-80 animate__animated animate__fadeIn">
                            <h2 className="text-2xl font-semibold mb-4">Sharing Your Information</h2>
                            <p className="text-gray-300 mb-4">
                                We do not share your personal information with third parties except in the following cases:
                            </p>
                            <ul className="list-disc list-inside text-gray-400 text-left">
                                <li>With your consent</li>
                                <li>To comply with legal obligations</li>
                                <li>To protect our rights and safety</li>
                            </ul>
                        </div>

                        {/* Card 5 */}
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700 w-full sm:w-80 animate__animated animate__fadeIn">
                            <h2 className="text-2xl font-semibold mb-4">Your Rights</h2>
                            <p className="text-gray-300 mb-4">
                                You have the right to:
                            </p>
                            <ul className="list-disc list-inside text-gray-400 text-left">
                                <li>Access your personal information</li>
                                <li>Request corrections to your data</li>
                                <li>Request the deletion of your data</li>
                            </ul>
                        </div>

                        {/* Card 6 */}
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700 w-full sm:w-80 animate__animated animate__fadeIn">
                            <h2 className="text-2xl font-semibold mb-4">Changes to This Privacy Policy</h2>
                            <p className="text-gray-300 mb-4">
                                We may update this Privacy Policy from time to time. Any changes will be reflected on this page.
                            </p>
                        </div>

                        {/* Card 7 */}
                        <div className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700 w-full sm:w-80 animate__animated animate__fadeIn">
                            <h2 className="text-2xl font-semibold mb-4">Contact Us</h2>
                            <p className="text-gray-300 mb-4">
                                If you have any questions about this Privacy Policy, please contact us at:
                                <br />
                                <span className="text-gray-400">virtualdiabetologist@gmail.com</span>
                            </p>
                        </div>
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
};

export default PrivacyPolicy;
