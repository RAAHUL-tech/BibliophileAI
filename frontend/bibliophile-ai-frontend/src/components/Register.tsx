import { useState, type FormEvent } from 'react'
import './SharedStyles.css'

interface RegisterProps {
  onRegisterSuccess: (jwtToken: string, sessionId: string) => void;
}

export default function Register({ onRegisterSuccess }: RegisterProps) {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState<string | null>(null)

  const handleRegister = async (e: FormEvent) => {
    e.preventDefault();
    setMessage(null);
    try {
      const res = await fetch('http://localhost:8080/api/v1/user/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, email, password }),
      });

      if (res.ok) {
        const data = await res.json();
        if (data.access_token && data.session_id) {
          onRegisterSuccess(data.access_token, data.session_id);
          setMessage(null);
          setUsername('');
          setEmail('');
          setPassword('');
        } else {
          setMessage('Registration successful, but missing token/session.');
        }
      } else {
        const error = await res.json();
        setMessage(`Error: ${error.detail || 'Failed to register'}`);
      }
    } catch {
      setMessage('Network error or server not reachable.');
    }
  };

  return (
    <form className="mb-3 animate__animated animate__fadeIn" onSubmit={handleRegister}>
      <input
        className="bib-input form-control mb-2"
        type="text"
        placeholder="Username"
        value={username}
        onChange={e => setUsername(e.target.value)}
        required
      />
      <input
        className="bib-input form-control mb-2"
        type="email"
        placeholder="Email"
        value={email}
        onChange={e => setEmail(e.target.value)}
        required
      />
      <input
        className="bib-input form-control mb-2"
        type="password"
        placeholder="Password"
        value={password}
        onChange={e => setPassword(e.target.value)}
        required
      />
      <button
        type="submit"
        className="bib-btn-primary w-100"
        style={{ borderRadius: '8px', padding: '0.6rem' }}
      >
        Register
      </button>
      {message && (
        <p
          className={`mt-2 fw-semibold text-center small`}
          style={{
            color: message.startsWith('Error') ? 'var(--bib-accent)' : '#28a745',
            animation: 'bib-fade-up 0.3s ease-out'
          }}
        >
          {message}
        </p>
      )}
    </form>
  )
}
