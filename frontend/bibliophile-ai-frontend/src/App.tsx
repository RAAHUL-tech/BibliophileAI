import { useState, useEffect } from 'react'
import { Routes, Route, Navigate, useNavigate } from 'react-router-dom'
import Register from './components/Register'
import Login from './components/Login'
import Homepage from './components/Homepage'
import UserOnboarding from "./components/UserOnboarding";
import { GoogleLogin } from '@react-oauth/google'

type AuthMode = 'login' | 'register'

export default function AppRoutes() {
  const [token, setToken] = useState<string | null>(() => {
  return sessionStorage.getItem('token');
});
const [sessionId, setSessionId] = useState<string | null>(() => {
  return sessionStorage.getItem('sessionId');
});

  const [authMode, setAuthMode] = useState<AuthMode>('login')
  const [googleMsg, setGoogleMsg] = useState<string | null>(null)
  const [isNewUser, setIsNewUser] = useState(false) // Track new user registration
  const navigate = useNavigate()

  useEffect(() => {
  if (!token) return;

  // If registering a new user, always go to onboarding flow.
  if (isNewUser) {
    navigate('/onboarding');
    return;
  }

  // Otherwise, check if the user has completed profile prefs.
  const fetchPreferences = async () => {
    try {
      const res = await fetch('http://localhost:8000/user/preferences', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const data = await res.json();
        if (!data || !data.genres || data.genres.length === 0) {
          navigate('/onboarding'); 
        } else {
          navigate('/home');
        }
      } else if (res.status === 404) {
        navigate('/onboarding');
      } else {
        navigate('/home');
      }
    } catch {
      navigate('/home');
    }
  };

  fetchPreferences();
}, [token, isNewUser, navigate]);

  const handleLoginSuccess = (jwtToken: string, sessionId: string) => {
    setToken(jwtToken)
    setSessionId(sessionId)
    sessionStorage.setItem('token', jwtToken);
    sessionStorage.setItem('sessionId', sessionId);
    setIsNewUser(false) // Mark returning user on login success
  }

  const handleLogout = () => {
    setToken(null)
    setSessionId(null)
    sessionStorage.removeItem('token');
    sessionStorage.removeItem('sessionId');
    setGoogleMsg(null)
    setIsNewUser(false)
    navigate('/')
  }

  const handleSwitch = (mode: AuthMode) => {
    setGoogleMsg(null)
    setAuthMode(mode)
  }

const handleGoogleAuth = async (credential: string) => {
 setGoogleMsg(null);
  try {
    const endpoint = authMode === 'register' ? '/google-register' : '/google-login';
    const res = await fetch(`http://localhost:8000${endpoint}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ credential }),
    });

    if (res.ok) {
      const data = await res.json();
      if (data.access_token && data.session_id) {
        // Store both token and session_id
        setToken(data.access_token);
        sessionStorage.setItem('token', data.access_token);
        setSessionId(data.session_id);
        sessionStorage.setItem('sessionId', data.session_id);

        // Mark as new user only on registration
        setIsNewUser(authMode === 'register');
      } else {
        setGoogleMsg('Failed to obtain access token or session.');
        if (authMode === 'register') setAuthMode('login');
      }
    } else {
      const err = await res.json();
      setGoogleMsg(`Error: ${err.detail || 'Google auth failed'}`);
    }
  } catch (error) {
    setGoogleMsg('Network error or server not reachable.');
  }
};


  const handleEmailRegisterSuccess = (jwtToken: string, sessionId: string) => {
    setToken(jwtToken)
    setSessionId(sessionId)
    sessionStorage.setItem('token', jwtToken);
    sessionStorage.setItem('sessionId', sessionId);
    setIsNewUser(true) // Mark new user on email registration success
  }

useEffect(() => {
  if (!sessionId) return;

  const handleUnload = () => {
    fetch("http://localhost:8000/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    });
  };

  window.addEventListener("beforeunload", handleUnload);
  return () => window.removeEventListener("beforeunload", handleUnload);
}, [sessionId]);

  return (
    <Routes>
      {!token ? (
        <Route
          path="/"
          element={
            <div className="container-fluid p-0" style={{ height: '100vh', overflow: 'hidden' }}>
              <div className="row g-0 h-100">
                {/* Left Side - Authentication - Exactly 50% */}
                <div className="col-lg-6 col-12 d-flex align-items-center justify-content-center bg-white position-relative" style={{ minHeight: '100vh' }}>
                  {/* Subtle background pattern for left side */}
                  <div 
                    className="position-absolute w-100 h-100 opacity-25"
                    style={{
                      backgroundImage: `url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23f8f9fa' fill-opacity='0.8' fill-rule='evenodd'%3E%3Ccircle cx='3' cy='3' r='3'/%3E%3Ccircle cx='13' cy='13' r='3'/%3E%3C/g%3E%3C/svg%3E")`,
                    }}
                  />
                  <div
                    className="card shadow-lg p-4 animate__animated animate__fadeIn position-relative mx-3"
                    style={{ 
                      maxWidth: '420px', 
                      width: '100%',
                      backdropFilter: 'blur(10px)',
                      background: 'rgba(255, 255, 255, 0.95)',
                      border: '1px solid rgba(255, 255, 255, 0.18)'
                    }}
                  >
                    <div className="text-center mb-4">
                      <h2 className="h3 text-primary mb-2 animate__animated animate__fadeInDown">
                        📚 BibliophileAI
                      </h2>
                      <p className="text-muted small animate__animated animate__fadeInDown">
                        Welcome back! Please sign in to continue
                      </p>
                    </div>
                    
                    {/* Register Form */}
                    {authMode === 'register' && (
                      <div
                        className="p-3 rounded-3 mb-3"
                        style={{
                          background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                          border: '1px solid rgba(0, 123, 255, 0.1)',
                          boxShadow: '0 2px 10px rgba(0, 123, 255, 0.08)'
                        }}
                      >
                        <h5 className="text-primary mb-3 text-center fw-bold h6">Create Account</h5>
                        <Register onRegisterSuccess={handleEmailRegisterSuccess} />
                        <div className="mt-3 text-center">
                          <GoogleLogin
                            onSuccess={(credentialResponse) => {
                              if (credentialResponse.credential) {
                                handleGoogleAuth(credentialResponse.credential)
                                setGoogleMsg('Processing Google registration...')
                              } else {
                                setGoogleMsg('Google registration failed: no credential')
                              }
                            }}
                            onError={() => setGoogleMsg('Google registration failed')}
                            theme="outline"
                            shape="circle"
                            width="280"
                            size="medium"
                            text="continue_with"
                          />
                          {googleMsg && (
                            <div className="mt-2 p-2 rounded bg-light border-start border-primary border-3">
                              <div className="small text-primary animate__animated animate__fadeIn">
                                {googleMsg}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Login Form */}
                    {authMode === 'login' && (
                      <div
                        className="p-3 rounded-3 mb-3"
                        style={{
                          background: 'linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%)',
                          border: '1px solid rgba(40, 167, 69, 0.1)',
                          boxShadow: '0 2px 10px rgba(40, 167, 69, 0.08)'
                        }}
                      >
                        <h5 className="text-success mb-3 text-center fw-bold h6">Sign In</h5>
                        <Login onLoginSuccess={handleLoginSuccess} />
                        <div className="mt-3 text-center">
                          <GoogleLogin
                            onSuccess={(credentialResponse) => {
                              if (credentialResponse.credential) {
                                handleGoogleAuth(credentialResponse.credential)
                                setGoogleMsg('Processing Google login...')
                              } else {
                                setGoogleMsg('Google login failed: no credential')
                              }
                            }}
                            onError={() => setGoogleMsg('Google login failed')}
                            theme="filled_black"
                            shape="pill"
                            width="280"
                            size="medium"
                            text="signin_with"
                          />
                          {googleMsg && (
                            <div className="mt-2 p-2 rounded bg-light border-start border-success border-3">
                              <div className="small text-success animate__animated animate__fadeIn">
                                {googleMsg}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                    
                    {/* Toggle Buttons */}
                    <div className="text-center">
                      {authMode === 'login' ? (
                        <button
                          className="btn btn-link text-decoration-none py-2 px-3 rounded-pill small"
                          style={{ 
                            fontWeight: 600,
                            transition: 'all 0.3s ease',
                            background: 'linear-gradient(135deg, rgba(0, 123, 255, 0.05), rgba(0, 123, 255, 0.1))'
                          }}
                          onClick={() => handleSwitch('register')}
                        >
                          New here? <span className="text-primary">Create an account</span>
                        </button>
                      ) : (
                        <button
                          className="btn btn-link text-decoration-none py-2 px-3 rounded-pill small"
                          style={{ 
                            fontWeight: 600,
                            transition: 'all 0.3s ease',
                            background: 'linear-gradient(135deg, rgba(40, 167, 69, 0.05), rgba(40, 167, 69, 0.1))'
                          }}
                          onClick={() => handleSwitch('login')}
                        >
                          Already have an account? <span className="text-success">Sign in</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Side - Hero Section - Exactly 50% */}
                <div
                  className="col-lg-6 d-none d-lg-flex align-items-center justify-content-center position-relative overflow-hidden"
                  style={{
                    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%)',
                    height: '100vh',
                  }}
                >
                  {/* Animated Background Elements */}
                  <div 
                    className="position-absolute w-100 h-100"
                    style={{
                      backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
                      opacity: 0.1,
                      animation: 'float 20s ease-in-out infinite'
                    }}
                  />
                  
                  {/* Floating Elements */}
                  <div className="position-absolute" style={{ top: '15%', left: '10%', animation: 'bounce 3s infinite' }}>
                    <div className="rounded-circle bg-white bg-opacity-10 p-2">
                      <span style={{ fontSize: '1.5rem' }}>🤖</span>
                    </div>
                  </div>
                  <div className="position-absolute" style={{ top: '25%', right: '15%', animation: 'bounce 3s infinite 1s' }}>
                    <div className="rounded-circle bg-white bg-opacity-10 p-2">
                      <span style={{ fontSize: '1.2rem' }}>⚡</span>
                    </div>
                  </div>
                  <div className="position-absolute" style={{ bottom: '20%', left: '15%', animation: 'bounce 3s infinite 2s' }}>
                    <div className="rounded-circle bg-white bg-opacity-10 p-2">
                      <span style={{ fontSize: '1.2rem' }}>🌐</span>
                    </div>
                  </div>

                  <div
                    className="text-center text-white px-4 animate__animated animate__fadeInRight position-relative"
                    style={{ zIndex: 2, maxWidth: '450px' }}
                  >
                    <div className="mb-4">
                      <h1
                        className="h2 fw-bold mb-3"
                        style={{ 
                          textShadow: '3px 3px 8px rgba(0,0,0,0.4)',
                          lineHeight: '1.2'
                        }}
                      >
                        Discover Your Next
                        <br />
                        <span
                          className="d-inline-block"
                          style={{
                            background: 'linear-gradient(45deg, #ffd700, #ffed4e, #fff)',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                            backgroundClip: 'text',
                            animation: 'shimmer 3s ease-in-out infinite'
                          }}
                        >
                          Favorite Book
                        </span>
                      </h1>
                      <p
                        className="mb-4"
                        style={{ 
                          textShadow: '2px 2px 6px rgba(0,0,0,0.4)',
                          lineHeight: '1.5',
                          fontSize: '1.1rem'
                        }}
                      >
                        AI-powered recommendations through advanced machine learning and social discovery
                      </p>
                    </div>

                    {/* Enhanced Features */}
                    <div className="mb-4">
                      <div 
                        className="d-flex align-items-center p-3 rounded-4 mb-3"
                        style={{
                          background: 'rgba(255, 255, 255, 0.1)',
                          backdropFilter: 'blur(10px)',
                          border: '1px solid rgba(255, 255, 255, 0.2)',
                        }}
                      >
                        <div className="rounded-circle bg-white bg-opacity-20 p-2 me-3 flex-shrink-0">
                          <span style={{ fontSize: '1.2rem' }}>🧠</span>
                        </div>
                        <div className="text-start">
                          <h6 className="mb-1 fw-bold small">Ensemble Machine Learning</h6>
                          <small className="opacity-85" style={{ fontSize: '0.8rem' }}>Advanced algorithms combining multiple ML models</small>
                        </div>
                      </div>
                      
                      <div 
                        className="d-flex align-items-center p-3 rounded-4 mb-3"
                        style={{
                          background: 'rgba(255, 255, 255, 0.1)',
                          backdropFilter: 'blur(10px)',
                          border: '1px solid rgba(255, 255, 255, 0.2)'
                        }}
                      >
                        <div className="rounded-circle bg-white bg-opacity-20 p-2 me-3 flex-shrink-0">
                          <span style={{ fontSize: '1.2rem' }}>🕸️</span>
                        </div>
                        <div className="text-start">
                          <h6 className="mb-1 fw-bold small">Graph Algorithms</h6>
                          <small className="opacity-85" style={{ fontSize: '0.8rem' }}>Social network analysis for book discovery</small>
                        </div>
                      </div>

                      <div 
                        className="d-flex align-items-center p-3 rounded-4"
                        style={{
                          background: 'rgba(255, 255, 255, 0.1)',
                          backdropFilter: 'blur(10px)',
                          border: '1px solid rgba(255, 255, 255, 0.2)'
                        }}
                      >
                        <div className="rounded-circle bg-white bg-opacity-20 p-2 me-3 flex-shrink-0">
                          <span style={{ fontSize: '1.2rem' }}>⚡</span>
                        </div>
                        <div className="text-start">
                          <h6 className="mb-1 fw-bold small">Real-time Streaming</h6>
                          <small className="opacity-85" style={{ fontSize: '0.8rem' }}>Instant updates as trends evolve</small>
                        </div>
                      </div>
                    </div>

                    {/* Enhanced Stats */}
                    <div className="row g-2 mb-4">
                      <div className="col-4">
                        <div 
                          className="p-3 rounded-4"
                          style={{
                            background: 'rgba(255, 255, 255, 0.15)',
                            backdropFilter: 'blur(15px)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
                          }}
                        >
                          <h4 className="fw-bold mb-1 text-warning">10M+</h4>
                          <small className="opacity-85 fw-medium" style={{ fontSize: '0.75rem' }}>Books Analyzed</small>
                        </div>
                      </div>
                      <div className="col-4">
                        <div 
                          className="p-3 rounded-4"
                          style={{
                            background: 'rgba(255, 255, 255, 0.15)',
                            backdropFilter: 'blur(15px)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
                          }}
                        >
                          <h4 className="fw-bold mb-1 text-warning">500K+</h4>
                          <small className="opacity-85 fw-medium" style={{ fontSize: '0.75rem' }}>Active Readers</small>
                        </div>
                      </div>
                      <div className="col-4">
                        <div 
                          className="p-3 rounded-4"
                          style={{
                            background: 'rgba(255, 255, 255, 0.15)',
                            backdropFilter: 'blur(15px)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)'
                          }}
                        >
                          <h4 className="fw-bold mb-1 text-warning">95%</h4>
                          <small className="opacity-85 fw-medium" style={{ fontSize: '0.75rem' }}>Accuracy Rate</small>
                        </div>
                      </div>
                    </div>

                    {/* Tech Stack */}
                    <div 
                      className="p-3 rounded-4"
                      style={{
                        background: 'rgba(0, 0, 0, 0.2)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(255, 255, 255, 0.1)'
                      }}
                    >
                      <h6 className="fw-bold mb-3 text-warning small">Built with Modern Architecture</h6>
                      <div className="d-flex flex-wrap justify-content-center gap-2">
                        <span className="badge bg-light text-dark px-2 py-1 rounded-pill fw-medium small">Kubernetes</span>
                        <span className="badge bg-light text-dark px-2 py-1 rounded-pill fw-medium small">Python</span>
                        <span className="badge bg-light text-dark px-2 py-1 rounded-pill fw-medium small">React</span>
                        <span className="badge bg-light text-dark px-2 py-1 rounded-pill fw-medium small">Microservices</span>
                      </div>
                      <p className="small opacity-75 mb-0 mt-2" style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.3)', fontSize: '0.8rem' }}>
                        Scalable • Intelligent • Social
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          }
        />
      ) : (
        <>
          <Route
            path="/onboarding"
            element={<UserOnboarding token={token!} onComplete={() => { setIsNewUser(false); navigate('/home') }} />}
          />
          <Route
            path="/home"
            element={<Homepage token={token!} onLogout={handleLogout} />}
          />
          <Route path="*" element={<Navigate to="/home" replace />} />
        </>
      )}
    </Routes>
  )
}