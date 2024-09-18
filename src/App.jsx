import Chatbot from './components/ChatBot/index'
import { useState } from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { useUser } from '@clerk/clerk-react'
import React from "react"
import Home from './components/Home'
import Header from './components/Header'
import Footer from './components/Footer'



function App() {

  return (
    <>
      <Header />
        <Home />
      <Footer />
    </>
  )
}

export default App
