import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import 'bootstrap/dist/css/bootstrap.min.css';
import 'animate.css';
import { GoogleOAuthProvider } from '@react-oauth/google';

const CLIENT_ID = "1080082180665-ucov4mb745ktpb8jqu8kivrg4h5bb4mb.apps.googleusercontent.com"

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <GoogleOAuthProvider clientId={CLIENT_ID}> 
    <App />
    </GoogleOAuthProvider>
  </StrictMode>,
)
