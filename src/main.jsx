import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx';
import './index.css';
import { ClerkProvider } from '@clerk/clerk-react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import Home from './components/Home/index.jsx';
import SignInPage from './components/SignInPage/index.jsx';
import Chatbot from './components/ChatBot/index.jsx';
import AboutUs from './components/AboutUs/index.jsx';
import NotFound from './components/NotFound/index.jsx'; // Import the NotFound component
import ContactUs from './components/ContactUs/index.jsx';
import OurServices from './components/OurServices/index.jsx';
import PrivacyPolicy from './components/PrivacyPolicy/index.jsx';
import API from './components/API/index.jsx';
import Retinopathy from './components/Retinopathy/index.jsx';
import DiabetesPredictor from './components/DiabetesPredictor/index.jsx';

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

if (!PUBLISHABLE_KEY) {
  throw new Error("Missing Publishable Key");
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />
  },
  {
    path: '/SignInPage',
    element: <SignInPage />
  },
  {
    path: '/ChatBot',
    element: <Chatbot />
  },
  {
    path: '/AboutUs',
    element: <AboutUs />
  },
  {
    path: '/ContactUs',
    element: <ContactUs />
  },
  {
    path: '/OurServices',
    element: <OurServices />
  },
  {
    path: '/PrivacyPolicy',
    element: <PrivacyPolicy />
  },
  {
    path: '/API',
    element: <API />
  },
  {
    path: '/Retinopathy',
    element: <Retinopathy />
  },
  {
    path: '/DiabetesPrediction',
    element: <DiabetesPredictor />
  },
  // Catch-all route for undefined paths
  {
    path: '*',
    element: <NotFound />
  }
]);

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY} afterSignOutUrl="/">
      <RouterProvider router={router} />
    </ClerkProvider>
  </React.StrictMode>
);
