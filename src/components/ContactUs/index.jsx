import React, { useState } from 'react';
import { MapPinIcon, EnvelopeIcon, PhoneIcon } from '@heroicons/react/24/outline';
import Header from '../Header';
import Footer from '../Footer';

const ContactUs = () => {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        contactNo: '',
        message: '',
    });

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData({
            ...formData,
            [name]: value,
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        // Simulate email submission
        const response = await fetch('/send-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData),
        });

        if (response.ok) {
            alert('Message sent!');
            setFormData({
                name: '',
                email: '',
                contactNo: '',
                message: '',
            });
        } else {
            alert('Failed to send message. Please try again.');
        }
    };

    return (
        <>
            <Header isInverted={true} />
            <div className="min-h-screen bg-black flex flex-col items-center justify-center p-32">
                <div className="flex items-center justify-center max-w-7xl w-full gap-10">
                    <div className="flex justify-center items-center">
                        <img
                            src="./contactUs.png"
                            alt="Contact Icon"
                            className="w-80 h-80 object-cover"
                        />
                    </div>
                    <div className="bg-gray-900 backdrop-blur-lg shadow-xl rounded-2xl p-8 w-full max-w-lg">
                        <h2 className="text-3xl font-semibold text-center text-white mb-6">Contact Us</h2>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label htmlFor="name" className="block text-lg text-white">Name</label>
                                <input
                                    type="text"
                                    id="name"
                                    name="name"
                                    value={formData.name}
                                    onChange={handleChange}
                                    required
                                    className="w-full p-2 rounded-md text-white bg-transparent bg-opacity-30 backdrop-blur-md border border-slate-600 focus:ring-2 focus:ring-purple-500"
                                    autoComplete='off'
                                />
                            </div>
                            <div>
                                <label htmlFor="email" className="block text-lg text-white">Email</label>
                                <input
                                    type="email"
                                    id="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    required
                                    className="w-full p-2 rounded-md text-white bg-transparent bg-opacity-30 backdrop-blur-md border border-slate-600 focus:ring-2 focus:ring-purple-500"
                                    autoComplete='off'
                                />
                            </div>
                            <div>
                                <label htmlFor="contactNo" className="block text-lg text-white">Contact No.</label>
                                <input
                                    type="tel"
                                    id="contactNo"
                                    name="contactNo"
                                    value={formData.contactNo}
                                    onChange={handleChange}
                                    required
                                    className="w-full p-2 rounded-md text-white bg-transparent bg-opacity-30 backdrop-blur-md border  border-slate-600 focus:ring-2 focus:ring-purple-500"
                                    autoComplete='off'
                                />
                            </div>
                            <div>
                                <label htmlFor="message" className="block text-lg text-white">Message</label>
                                <textarea
                                    id="message"
                                    name="message"
                                    value={formData.message}
                                    onChange={handleChange}
                                    required
                                    className="w-full p-2 rounded-md text-white bg-transparent bg-opacity-30 backdrop-blur-md border  border-slate-600 focus:ring-2 focus:ring-red-400"
                                    autoComplete='off'
                                ></textarea>
                            </div>
                            <button
                                type="submit"
                                className="w-full p-3 text-white bg-gradient-to-r from-purple-500 to-blue-500 rounded-md shadow-md hover:scale-105 hover:shadow-lg duration-300"

                            >
                                Send Message
                            </button>
                        </form>
                    </div>
                </div>

                <div className="w-full mt-16 p-8 bg-gray-900 rounded-xl shadow-lg">
                    <div className="text-white space-y-6">
                        <h3 className="text-2xl font-semibold text-center">Get in touch with us</h3>
                        <div className="flex justify-center space-x-8">
                            <div className="flex items-center space-x-4">
                                <EnvelopeIcon className="h-6 w-6 text-purple-500" />
                                <p>virtualdiabetologist@gmail.com</p>
                            </div>
                            <div className="flex items-center space-x-4">
                                <PhoneIcon className="h-6 w-6 text-purple-500" />
                                <p>+92 333 0011213</p>
                            </div>
                            <div className="flex items-center space-x-4">
                                <MapPinIcon className="h-6 w-6 text-purple-500" />
                                <p>Sector E8, Islamabad, Pakistan</p>
                            </div>
                        </div>
                        <div className="flex justify-center">
                            <iframe
                                src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3318.504290402195!2d73.04337027444397!3d33.70203188069927!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x38dfbeaf837e7bff%3A0x84f120dfe43df3a!2sBahria%20University%20Islamabad!5e0!3m2!1sen!2s!4v1697282296407!5m2!1sen!2s" // Replace with actual map link
                                className="w-full max-w-2xl h-64 rounded-lg"
                                allowFullScreen
                                loading="lazy"
                            ></iframe>
                        </div>
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
};

export default ContactUs;
