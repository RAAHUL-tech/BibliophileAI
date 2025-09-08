import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Register from './components/Register'
import Login from './components/Login'
import Homepage from './components/Homepage'

function App() {
  const [token, setToken] = useState<string | null>(null)

  const handleLoginSuccess = (jwtToken: string) => setToken(jwtToken)
  const handleLogout = () => setToken(null)

  return (
    <Router>
      <div className="container-fluid min-vh-100 bg-light d-flex flex-column justify-content-center align-items-center">
        <Routes>
          {!token ? (
            <>
              <Route
                path="/"
                element={
                  <div className="card shadow-lg p-4" style={{ maxWidth: 450, width: '100%' }}>
                    <h1 className="display-5 text-center mb-3 animate__animated animate__fadeInDown">BibliophileAI</h1>
                    <h6 className="text-secondary text-center mb-4">Social &amp; Intelligent Book Recommendation System</h6>
                    <h2 className="h5 mb-2 text-primary animate__animated animate__fadeInLeft">Register</h2>
                    <Register />
                    <hr className="my-4" />
                    <h2 className="h5 mb-2 text-success animate__animated animate__fadeInRight">Login</h2>
                    <Login onLoginSuccess={handleLoginSuccess} />
                  </div>
                }
              />
              <Route path="*" element={<Navigate to="/" replace />} />
            </>
          ) : (
            <>
              <Route path="/home" element={<Homepage token={token} onLogout={handleLogout} />} />
              <Route path="*" element={<Navigate to="/home" replace />} />
            </>
          )}
        </Routes>
      </div>
    </Router>
  )
}
export default App
