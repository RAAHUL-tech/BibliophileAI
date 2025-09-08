import type { FC } from 'react'

interface HomepageProps {
  token: string
  onLogout: () => void
}

const Homepage: FC<HomepageProps> = ({ token, onLogout }) => (
  <div className="card shadow-lg p-4 animate__animated animate__fadeInDown" style={{ maxWidth: 450, width: '100%', margin: '40px auto' }}>
    <h1 className="text-center mb-4 text-primary">Welcome to BibliophileAI</h1>
    <p className="fw-semibold text-center mb-3">Your JWT token:</p>
    <pre className="token-display bg-light px-2 py-2 rounded text-break small mb-4">{token}</pre>
    <button
      className="btn btn-danger w-100"
      onClick={() => {
        onLogout()
        window.location.href = '/'
      }}
    >
      Logout
    </button>
  </div>
)
export default Homepage
