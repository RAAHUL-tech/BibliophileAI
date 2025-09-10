import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Register from './components/Register'
import Login from './components/Login'
import Homepage from './components/Homepage'
import { GoogleLogin } from '@react-oauth/google'

type AuthMode = 'login' | 'register'

function App() {
  const [token, setToken] = useState<string | null>(null)
  const [authMode, setAuthMode] = useState<AuthMode>('login')
  const [googleMsg, setGoogleMsg] = useState<string | null>(null)

  const handleLoginSuccess = (jwtToken: string) => setToken(jwtToken)
  const handleLogout = () => setToken(null)
  const handleSwitch = (mode: AuthMode) => {
    setGoogleMsg(null)
    setAuthMode(mode)
  }

  // send google credential to backend and get JWT token
  const handleGoogleAuth = async (credential: string) => {
    setGoogleMsg(null)
    try {
      console.log("Sending credential to backend:", credential);
      const endpoint = authMode === 'register' ? '/google-register' : '/google-login'
      const res = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential }),
      })
      if (res.ok) {
        const data = await res.json()
        if (data.access_token) {
          handleLoginSuccess(data.access_token)
        } else {
          console.error("No access_token in response:", data);
          setGoogleMsg('Failed to obtain access token from server.')
        }
      } else {
        console.error("Google auth failed with status:", res.status);
        const err = await res.json()
        setGoogleMsg(`Error: ${err.detail || 'Google auth failed'}`)
      }
    } catch {
      setGoogleMsg('Network error or server not reachable.')
    }
  }

  return (
    <Router>
      <div className="min-vh-100 bg-light">
        <Routes>
          {!token ? (
            <>
              <Route
                path="/"
                element={
                  <div className="container-fluid px-0">
                    <div className="row g-0 min-vh-100">
                      {/* Left Side - Authentication */}
                      <div className="col-md-6 d-flex align-items-center justify-content-center bg-white">
                        <div
                          className="card shadow-lg p-4 animate__animated animate__fadeIn"
                          style={{
                            maxWidth: 450,
                            width: '100%',
                            margin: '2rem',
                          }}
                        >
                          <div className="text-center mb-4">
                            <h2 className="display-6 text-primary mb-2 animate__animated animate__fadeInDown">
                              📚 BibliophileAI
                            </h2>
                            <p className="text-muted animate__animated animate__fadeInDown">
                              Welcome back! Please sign in to continue
                            </p>
                          </div>

                          <div className="row g-3">
                            {/* Register */}
                            <div className="col-12">
                              <div
                                className={`p-3 bg-light rounded-3 ${
                                  authMode === 'login' ? 'opacity-50' : ''
                                }`}
                                style={{
                                  filter: authMode === 'login' ? 'blur(1px)' : 'none',
                                  transition: 'filter .35s, opacity .35s',
                                  display: authMode === 'register' ? 'block' : 'none',
                                }}
                              >
                                <h5 className="text-primary mb-3 text-center">Create Account</h5>
                                <Register />
                                <div className="my-3 text-center">
                                  <GoogleLogin
                                    onSuccess={credentialResponse => {
                                      if (credentialResponse.credential) {
                                        console.log("Google credential:", credentialResponse.credential);
                                        handleGoogleAuth(credentialResponse.credential)
                                        setGoogleMsg("Processing Google registration...")
                                      } else {
                                        setGoogleMsg("Google registration failed: no credential")
                                      }
                                    }}
                                    onError={() => setGoogleMsg('Google registration failed')}
                                    theme="outline"
                                    shape="circle"
                                    width="300"
                                    size="large"
                                    text="continue_with"
                                  />
                                  {authMode === 'register' && googleMsg && (
                                    <div className="mt-2 fw-semibold text-success animate__animated animate__fadeIn">
                                      {googleMsg}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>

                            {/* Login */}
                            <div className="col-12">
                              <div
                                className={`p-3 bg-light rounded-3 ${
                                  authMode === 'register' ? 'opacity-50' : ''
                                }`}
                                style={{
                                  filter: authMode === 'register' ? 'blur(1px)' : 'none',
                                  transition: 'filter .35s, opacity .35s',
                                  display: authMode === 'login' ? 'block' : 'none',
                                }}
                              >
                                <h5 className="text-success mb-3 text-center">Sign In</h5>
                                <Login onLoginSuccess={handleLoginSuccess} />
                                <div className="my-3 text-center">
                                  <GoogleLogin
                                    onSuccess={credentialResponse => {
                                      if (credentialResponse.credential) {
                                        handleGoogleAuth(credentialResponse.credential)
                                        setGoogleMsg("Processing Google login...")
                                      } else {
                                        setGoogleMsg("Google login failed: no credential")
                                      }
                                    }}
                                    onError={() => setGoogleMsg('Google login failed')}
                                    theme="filled_black"
                                    shape="pill"
                                    width="300"
                                    size="large"
                                    text="signin_with"
                                  />
                                  {authMode === 'login' && googleMsg && (
                                    <div className="mt-2 fw-semibold text-success animate__animated animate__fadeIn">
                                      {googleMsg}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Toggle Buttons */}
                          <div className="text-center mt-3">
                            {authMode === 'login' ? (
                              <button
                                className="btn btn-link text-decoration-none"
                                style={{ fontWeight: 600 }}
                                onClick={() => handleSwitch('register')}
                              >
                                New here? <span className="text-primary">Create an account</span>
                              </button>
                            ) : (
                              <button
                                className="btn btn-link text-decoration-none"
                                style={{ fontWeight: 600 }}
                                onClick={() => handleSwitch('login')}
                              >
                                Already have an account? <span className="text-success">Sign in</span>
                              </button>
                            )}
                          </div>
                        </div>
                      </div>

                      {/* Right Side - Hero Section */}
                      <div className="col-md-6 d-flex align-items-center justify-content-center position-relative" 
                           style={{ 
                             background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                             minHeight: '100vh'
                           }}>
                        {/* Background Pattern */}
                        <div className="position-absolute w-100 h-100" 
                             style={{
                               backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
                               opacity: 0.1
                             }}>
                        </div>
                        
                        <div className="text-center text-white p-5 animate__animated animate__fadeInRight" style={{ zIndex: 1 }}>
                          <div className="mb-4">
                            <h1 className="display-4 fw-bold mb-3" style={{ textShadow: '2px 2px 4px rgba(0,0,0,0.3)' }}>
                              Discover Your Next
                              <br />
                              <span style={{ 
                                background: 'linear-gradient(45deg, #ffd700, #ffed4e)',
                                WebkitBackgroundClip: 'text',
                                WebkitTextFillColor: 'transparent',
                                backgroundClip: 'text'
                              }}>
                                Favorite Book
                              </span>
                            </h1>
                            <p className="lead mb-4" style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.3)' }}>
                              AI-powered recommendations that understand your reading preferences
                            </p>
                          </div>

                          {/* Features */}
                          <div className="row g-4 mb-5">
                            <div className="col-12">
                              <div className="d-flex align-items-center justify-content-center mb-3">
                                <div className="rounded-circle bg-white bg-opacity-20 p-3 me-3">
                                  <span className="fs-3">🤖</span>
                                </div>
                                <div className="text-start">
                                  <h6 className="mb-1 fw-bold">Smart AI Recommendations</h6>
                                  <small className="opacity-75">Machine learning algorithms analyze your reading history</small>
                                </div>
                              </div>
                            </div>
                            
                            <div className="col-12">
                              <div className="d-flex align-items-center justify-content-center mb-3">
                                <div className="rounded-circle bg-white bg-opacity-20 p-3 me-3">
                                  <span className="fs-3">👥</span>
                                </div>
                                <div className="text-start">
                                  <h6 className="mb-1 fw-bold">Social Discovery</h6>
                                  <small className="opacity-75">Connect with readers who share your taste</small>
                                </div>
                              </div>
                            </div>

                            <div className="col-12">
                              <div className="d-flex align-items-center justify-content-center mb-3">
                                <div className="rounded-circle bg-white bg-opacity-20 p-3 me-3">
                                  <span className="fs-3">⚡</span>
                                </div>
                                <div className="text-start">
                                  <h6 className="mb-1 fw-bold">Real-time Updates</h6>
                                  <small className="opacity-75">Get instant recommendations as trends change</small>
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Stats */}
                          <div className="row g-3 mb-4">
                            <div className="col-4">
                              <div className="bg-white bg-opacity-10 rounded-3 p-3 backdrop-blur">
                                <h4 className="fw-bold mb-1">10M+</h4>
                                <small className="opacity-75">Books</small>
                              </div>
                            </div>
                            <div className="col-4">
                              <div className="bg-white bg-opacity-10 rounded-3 p-3 backdrop-blur">
                                <h4 className="fw-bold mb-1">500K+</h4>
                                <small className="opacity-75">Users</small>
                              </div>
                            </div>
                            <div className="col-4">
                              <div className="bg-white bg-opacity-10 rounded-3 p-3 backdrop-blur">
                                <h4 className="fw-bold mb-1">95%</h4>
                                <small className="opacity-75">Accuracy</small>
                              </div>
                            </div>
                          </div>

                          <div className="mt-4">
                            <p className="small opacity-75 mb-0" style={{ textShadow: '1px 1px 2px rgba(0,0,0,0.3)' }}>
                              Powered by Kubernetes • Machine Learning • Graph Algorithms
                            </p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </>
          ) : (
            <>
              <Route 
                path="/home" 
                element={<Homepage token={token} onLogout={handleLogout} />} 
              />
              <Route path="*" element={<Navigate to="/home" replace />} />
            </>
          )}
        </Routes>
      </div>
    </Router>
  )
}

export default App