import React from 'react';
import { FaDatabase, FaRobot, FaUtensils, FaHeartbeat, FaDumbbell, FaHistory } from 'react-icons/fa';
import Header from '../Header';
import Footer from '../Footer';

const services = [
    {
        title: 'API Integration Services',
        description: 'We provide API services for developers who wish to integrate our system into theirs using third-party integrations.',
        icon: <FaDatabase className="w-12 h-12 text-purple-500" />,
    },
    {
        title: 'Custom Chatbot Solutions',
        description: 'Our virtual diabetologist offers the most accurate responses and aims to assist users in managing diabetes effectively.',
        icon: <FaRobot className="w-12 h-12 text-purple-500" />,
    },
    {
        title: 'Personalized Meal Planning',
        description: 'Our system allows users to plan meals based on their diabetes severity, always maintaining user-specific context.',
        icon: <FaUtensils className="w-12 h-12 text-purple-500" />,
    },
    {
        title: 'User-Specific Context',
        description: 'Our system maintains a user-specific context by gathering necessary vital information from the user.',
        icon: <FaHeartbeat className="w-12 h-12 text-purple-500" />,
    },
    {
        title: 'Customized Workout Plans',
        description: 'The virtual diabetologist offers tailored workout plans, considering the medical conditions of the user.',
        icon: <FaDumbbell className="w-12 h-12 text-purple-500" />,
    },
    {
        title: 'Chat History Management',
        description: 'Our system maintains a history stack necessary for preserving conversation context for the language model.',
        icon: <FaHistory className="w-12 h-12 text-purple-500" />,
    },
];

const OurServices = () => {
    return (
        <>
            <Header isInverted={true} />
            <div className="bg-gray-900 text-white py-16 px-4 pt-32">
                <div className="max-w-7xl mx-auto text-center">
                    <h2 className="text-4xl font-bold mb-4">
                        OUR <span className="text-purple-500">SERVICES</span>
                    </h2>
                    <p className="mb-12 text-gray-400">
                        We take pride in what we build
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
                        {services.map((service, index) => (
                            <div
                                key={index}
                                className="bg-gray-800 p-6 rounded-lg shadow-md transition-transform duration-300 hover:scale-105 hover:bg-gray-700"
                            >
                                <div className="flex justify-center mb-4">
                                    {service.icon}
                                </div>
                                <h3 className="text-2xl font-semibold mb-2">{service.title}</h3>
                                <p className="text-gray-300">{service.description}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
            <Footer />
        </>
    );
};

export default OurServices;
