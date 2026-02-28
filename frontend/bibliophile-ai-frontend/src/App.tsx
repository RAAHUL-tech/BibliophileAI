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
  const [isNewUser, setIsNewUser] = useState(false)
  const navigate = useNavigate()

  useEffect(() => {
    if (!token) return;

    if (isNewUser) {
      navigate('/onboarding');
      return;
    }

    const fetchPreferences = async () => {
      try {
        const res = await fetch('http://localhost:8080/api/v1/user/preferences', {
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
    setIsNewUser(false)
  }

  const handleLogout = async () => {
    try {
      console.log("Logout called with:", { sessionId, token });

      if (sessionId && token) {
        const res = await fetch("http://localhost:8080/api/v1/user/logout", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token}`,
          },
          body: JSON.stringify({ session_id: sessionId }),
        });
        console.log("Logout response status:", res.status);
        if (!res.ok) {
          const text = await res.text();
          console.error("Logout failed:", text);
        } else {
          console.log("Logout succeeded");
        }
      } else {
        console.warn("Missing sessionId or token, skip logout API call");
      }
    } catch (error) {
      console.error("Logout request failed:", error);
    }

    setToken(null);
    setSessionId(null);
    sessionStorage.removeItem("token");
    sessionStorage.removeItem("sessionId");
    setGoogleMsg(null);
    setIsNewUser(false);
    navigate("/");
  };

  const handleSwitch = (mode: AuthMode) => {
    setGoogleMsg(null)
    setAuthMode(mode)
  }

  const handleGoogleAuth = async (credential: string) => {
    setGoogleMsg(null);
    try {
      const endpoint = authMode === 'register' ? '/google-register' : '/google-login';
      const res = await fetch(`http://localhost:8080/api/v1/user${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ credential }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.access_token && data.session_id) {
          setToken(data.access_token);
          sessionStorage.setItem('token', data.access_token);
          setSessionId(data.session_id);
          sessionStorage.setItem('sessionId', data.session_id);
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
    setIsNewUser(true)
  }

  return (
    <Routes>
      {!token ? (
        <Route
          path="/"
          element={
            <div
              className="container-fluid p-0"
              style={{ height: '100vh', overflow: 'hidden', background: 'var(--bib-bg)' }}
            >
              <div className="row g-0 h-100">
                {/* Left Side — Auth panel, theme-aware */}
                <div
                  className="col-lg-6 col-12 d-flex align-items-center justify-content-center position-relative"
                  style={{
                    minHeight: '100vh',
                    background: 'var(--bib-bg)',
                    transition: 'background 0.35s ease'
                  }}
                >
                  {/* Subtle background pattern */}
                  <div
                    className="position-absolute w-100 h-100"
                    style={{
                      backgroundImage: `url("data:image/svg+xml,%3Csvg width='20' height='20' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23667eea' fill-opacity='0.04' fill-rule='evenodd'%3E%3Ccircle cx='3' cy='3' r='3'/%3E%3Ccircle cx='13' cy='13' r='3'/%3E%3C/g%3E%3C/svg%3E")`,
                      pointerEvents: 'none'
                    }}
                  />

                  <div
                    className="card shadow-lg p-4 animate__animated animate__fadeIn position-relative mx-3"
                    style={{
                      maxWidth: '420px',
                      width: '100%',
                      backdropFilter: 'blur(10px)',
                      background: 'var(--bib-bg-elevated)',
                      border: '1px solid var(--bib-border)',
                      borderRadius: '16px',
                      transition: 'background 0.35s ease, border-color 0.35s ease'
                    }}
                  >
                    <div className="text-center mb-4">
                      <h2
                        className="h3 mb-2 animate__animated animate__fadeInDown"
                        style={{ color: 'var(--bib-accent)', fontWeight: 800, letterSpacing: '-0.02em' }}
                      >
                        📚 BibliophileAI
                      </h2>
                      <p
                        className="small animate__animated animate__fadeInDown"
                        style={{ color: 'var(--bib-text-muted)', marginBottom: 0 }}
                      >
                        Welcome back! Please sign in to continue
                      </p>
                    </div>

                    {/* Register Form */}
                    {authMode === 'register' && (
                      <div
                        className="p-3 rounded-3 mb-3"
                        style={{
                          background: 'var(--bib-card-header)',
                          border: '1px solid var(--bib-border)',
                          transition: 'background 0.35s ease, border-color 0.35s ease'
                        }}
                      >
                        <h5
                          className="mb-3 text-center fw-bold h6"
                          style={{ color: 'var(--bib-accent)' }}
                        >
                          Create Account
                        </h5>
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
                            <div
                              className="mt-2 p-2 rounded animate__animated animate__fadeIn"
                              style={{
                                background: 'var(--bib-tab-active-bg)',
                                border: '1px solid var(--bib-border)',
                                borderLeft: '3px solid var(--bib-accent)'
                              }}
                            >
                              <div className="small" style={{ color: 'var(--bib-accent)' }}>
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
                          background: 'var(--bib-card-header)',
                          border: '1px solid var(--bib-border)',
                          transition: 'background 0.35s ease, border-color 0.35s ease'
                        }}
                      >
                        <h5
                          className="mb-3 text-center fw-bold h6"
                          style={{ color: '#28a745' }}
                        >
                          Sign In
                        </h5>
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
                            <div
                              className="mt-2 p-2 rounded animate__animated animate__fadeIn"
                              style={{
                                background: 'rgba(40, 167, 69, 0.1)',
                                border: '1px solid rgba(40, 167, 69, 0.2)',
                                borderLeft: '3px solid #28a745'
                              }}
                            >
                              <div className="small" style={{ color: '#28a745' }}>
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
                            color: 'var(--bib-text-muted)',
                            background: 'var(--bib-btn-secondary-bg)'
                          }}
                          onClick={() => handleSwitch('register')}
                        >
                          New here?{' '}
                          <span style={{ color: 'var(--bib-accent)' }}>Create an account</span>
                        </button>
                      ) : (
                        <button
                          className="btn btn-link text-decoration-none py-2 px-3 rounded-pill small"
                          style={{
                            fontWeight: 600,
                            transition: 'all 0.3s ease',
                            color: 'var(--bib-text-muted)',
                            background: 'var(--bib-btn-secondary-bg)'
                          }}
                          onClick={() => handleSwitch('login')}
                        >
                          Already have an account?{' '}
                          <span style={{ color: '#28a745' }}>Sign in</span>
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Right Side — Book Community Hero */}
                <div
                  className="col-lg-6 d-none d-lg-flex align-items-center justify-content-center position-relative overflow-hidden"
                  style={{
                    background: 'linear-gradient(160deg, #0a1628 0%, #0f2040 45%, #162d50 100%)',
                    height: '100vh',
                  }}
                >
                  {/* Subtle dot grid */}
                  <div
                    className="position-absolute w-100 h-100"
                    style={{
                      backgroundImage: 'radial-gradient(circle, rgba(255,255,255,0.055) 1px, transparent 1px)',
                      backgroundSize: '26px 26px',
                      pointerEvents: 'none',
                    }}
                  />

                  {/* Warm amber glow top-right */}
                  <div
                    className="position-absolute"
                    style={{
                      width: '380px',
                      height: '380px',
                      background: 'radial-gradient(circle, rgba(229,152,60,0.13) 0%, transparent 70%)',
                      top: '-60px',
                      right: '-80px',
                      pointerEvents: 'none',
                    }}
                  />

                  {/* Red accent glow bottom-left */}
                  <div
                    className="position-absolute"
                    style={{
                      width: '300px',
                      height: '300px',
                      background: 'radial-gradient(circle, rgba(229,9,20,0.09) 0%, transparent 70%)',
                      bottom: '-40px',
                      left: '-60px',
                      pointerEvents: 'none',
                    }}
                  />

                  {/* Decorative floating book icons */}
                  <div className="position-absolute" style={{ top: '10%', left: '7%', opacity: 0.18, fontSize: '3rem', animation: 'bounce 5s ease-in-out infinite' }}>📚</div>
                  <div className="position-absolute" style={{ bottom: '12%', right: '6%', opacity: 0.14, fontSize: '2.4rem', animation: 'bounce 5s ease-in-out infinite 1.8s' }}>📖</div>
                  <div className="position-absolute" style={{ top: '55%', left: '4%', opacity: 0.11, fontSize: '2rem', animation: 'bounce 5s ease-in-out infinite 3.2s' }}>🔖</div>

                  <div
                    className="text-center px-5 position-relative animate__animated animate__fadeInRight"
                    style={{ zIndex: 2, maxWidth: '460px' }}
                  >
                    {/* Icon mark */}
                    <div style={{ fontSize: '3.2rem', marginBottom: '0.75rem' }}>📚</div>

                    {/* Headline */}
                    <h1 className="fw-bold mb-2" style={{ color: '#fff', fontSize: '1.9rem', lineHeight: 1.25, letterSpacing: '-0.02em' }}>
                      Your Reading Journey,
                      <br />
                      <span
                        style={{
                          background: 'linear-gradient(90deg, #f59e0b, #fbbf24, #f59e0b)',
                          backgroundSize: '200% 100%',
                          WebkitBackgroundClip: 'text',
                          WebkitTextFillColor: 'transparent',
                          backgroundClip: 'text',
                          animation: 'shimmer 3s ease-in-out infinite',
                        }}
                      >
                        Reimagined
                      </span>
                    </h1>

                    {/* Quote */}
                    <p style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.85rem', fontStyle: 'italic', marginBottom: '1.75rem', lineHeight: 1.5 }}>
                      "A reader lives a thousand lives before he dies."
                      <span style={{ display: 'block', fontStyle: 'normal', color: 'rgba(255,255,255,0.3)', fontSize: '0.75rem', marginTop: '0.3rem' }}>— George R.R. Martin</span>
                    </p>

                    {/* Feature list */}
                    {[
                      { icon: '🤝', title: 'Connect with Friends', desc: 'Follow readers you trust, share picks, and discover books through your social circle' },
                      { icon: '✨', title: 'Personalized For You', desc: 'AI that learns your taste across genres — from timeless classics to new releases' },
                      { icon: '🔖', title: 'Build Your Library', desc: 'Track what you\'ve read, what\'s on your wishlist, and your reading progress' },
                      { icon: '🌍', title: 'Join a Global Community', desc: 'Thousands of passionate readers discovering their next favourite book every day' },
                    ].map((feature, i) => (
                      <div
                        key={i}
                        className="d-flex align-items-start p-3 rounded-4 mb-2 text-start"
                        style={{
                          background: 'rgba(255,255,255,0.04)',
                          border: '1px solid rgba(255,255,255,0.07)',
                          backdropFilter: 'blur(8px)',
                          animation: `fadeInUp 0.45s ease-out ${0.08 * i}s both`,
                        }}
                      >
                        <span style={{ fontSize: '1.25rem', marginRight: '0.75rem', marginTop: '0.05rem', flexShrink: 0 }}>{feature.icon}</span>
                        <div>
                          <div style={{ color: '#fff', fontWeight: 700, fontSize: '0.88rem', marginBottom: '0.2rem' }}>{feature.title}</div>
                          <div style={{ color: 'rgba(255,255,255,0.48)', fontSize: '0.76rem', lineHeight: 1.45 }}>{feature.desc}</div>
                        </div>
                      </div>
                    ))}

                    {/* Stats strip */}
                    <div className="d-flex justify-content-center gap-4 mt-3 pt-2" style={{ borderTop: '1px solid rgba(255,255,255,0.07)' }}>
                      {[
                        { value: '50K+', label: 'Books' },
                        { value: '99%', label: 'Personalized' },
                        { value: '∞', label: 'Adventures' },
                      ].map((stat, i) => (
                        <div key={i} className="text-center">
                          <div style={{ color: '#f59e0b', fontWeight: 800, fontSize: '1.15rem' }}>{stat.value}</div>
                          <div style={{ color: 'rgba(255,255,255,0.38)', fontSize: '0.7rem', marginTop: '0.1rem' }}>{stat.label}</div>
                        </div>
                      ))}
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
