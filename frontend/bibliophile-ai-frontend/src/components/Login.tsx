import { useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'

interface LoginProps {
  onLoginSuccess: (token: string, sessionId: string) => void
}

export default function Login({ onLoginSuccess }: LoginProps) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const navigate = useNavigate()

  const handleLogin = async (e: FormEvent) => {
    e.preventDefault()
    setError(null)
    const formData = new URLSearchParams()
    formData.append('username', username)
    formData.append('password', password)

    try {
      const res = await fetch('http://localhost:8000/api/v1/user/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData.toString(),
      })
      if (res.ok) {
        const data = await res.json()
        onLoginSuccess(data.access_token, data.session_id)
        setUsername('')
        setPassword('')
        navigate('/home')
      } else {
        setError('Invalid username or password.')
      }
    } catch {
      setError('Network error or server not reachable.')
    }
  }

  return (
    <form className="mb-2 animate__animated animate__fadeIn" onSubmit={handleLogin}>
      <input
        className="form-control mb-2"
        type="text"
        placeholder="Username"
        value={username}
        onChange={e => setUsername(e.target.value)}
        required
      />
      <input
        className="form-control mb-2"
        type="password"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        required
      />
      <button type="submit" className="btn btn-success w-100">Login</button>
      {error && <p className="mt-2 text-danger fw-semibold text-center">{error}</p>}
    </form>
  )
}
