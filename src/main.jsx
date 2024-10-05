import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { ClerkProvider } from '@clerk/clerk-react'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import Home from './components/Home/index.jsx'
import SignInPage from './components/SignInPage/index.jsx'
import Chatbot from './components/ChatBot/index.jsx'
import AboutUs from './components/AboutUs/index.jsx'

const PUBLISHABLE_KEY = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY

if (!PUBLISHABLE_KEY) {
  throw new Error("Missing Publishable Key")
}

const router = createBrowserRouter([
  {
    path: '/',
    element:<App />
  },
  {
    path: '/SignInPage',
    element:<SignInPage />
  },
  {
    path: '/ChatBot',
    element: <Chatbot />
  },
  {
    path: '/AboutUs',
    element: <AboutUs />
  }
])

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ClerkProvider publishableKey={PUBLISHABLE_KEY} afterSignOutUrl="/">
      <RouterProvider router={router} />
    </ClerkProvider>
  </React.StrictMode>,
)
