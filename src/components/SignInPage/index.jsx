import { SignIn } from '@clerk/clerk-react'
import { useEffect } from 'react';
import React from 'react'
import './SignInPage.css'

function SignInPage() {



  return (
    <>
    <div className='main'>
      <div className='logo-div'>
      <img src="./logo.svg" alt="Logo" width={200} height={200} />
      </div>
      <div className='sign-in-main'>
        <SignIn />
      </div>
    </div>
    </>
  )
}

export default SignInPage