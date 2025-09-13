import { useState, type FormEvent } from 'react'

interface RegisterProps {
  onRegisterSuccess: (jwtToken: string) => void
}

export default function Register({ onRegisterSuccess }: RegisterProps) {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState<string | null>(null)

  const handleRegister = async (e: FormEvent) => {
    e.preventDefault()
    setMessage(null)
    try {
      const res = await fetch('http://localhost:8000/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      })
      if (res.ok) {
        // Assuming backend returns access_token on successful register
        const data = await res.json()
        if (data.access_token) {
          onRegisterSuccess(data.access_token) // Notify parent of successful registration + token
          setMessage(null)
          setUsername('')
          setEmail('')
          setPassword('')
        } else {
          setMessage('User registered successfully! Please login.')
        }
      } else {
        const error = await res.json()
        setMessage(`Error: ${error.detail || 'Failed to register'}`)
      }
    } catch {
      setMessage('Network error or server not reachable.')
    }
  }

  return (
    <form className="mb-3 animate__animated animate__fadeIn" onSubmit={handleRegister}>
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
        type="email"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
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
      <button type="submit" className="btn btn-primary w-100">
        Register
      </button>
      {message && (
        <p className={`mt-2 fw-semibold text-center ${message.startsWith('Error') ? 'text-danger' : 'text-success'}`}>
          {message}
        </p>
      )}
    </form>
  )
}
