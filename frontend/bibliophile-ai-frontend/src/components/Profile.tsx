import { useEffect, useState } from "react"

interface ProfileProps {
  token: string
  onClose: () => void
}

interface UserProfile {
  username: string
  email: string
  age: number | null
  pincode: string | null
}

export default function Profile({ token, onClose }: ProfileProps) {
  const [profile, setProfile] = useState<UserProfile | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchProfile = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch("http://localhost:8000/user/profile", {
          headers: { Authorization: `Bearer ${token}` }
        })
        if (res.ok) {
          setProfile(await res.json())
        } else {
          setError("Failed to load profile")
        }
      } catch {
        setError("Network error while loading profile")
      } finally {
        setLoading(false)
      }
    }
    fetchProfile()
  }, [token])

  return (
    <div className="container my-5">
      <div className="card shadow-lg mx-auto" style={{ maxWidth: 450 }}>
        <div className="card-header text-center bg-primary text-white">
          <h4>User Profile</h4>
        </div>
        <div className="card-body">
          {loading ? (
            <p>Loading...</p>
          ) : error ? (
            <p className="text-danger">{error}</p>
          ) : profile ? (
            <>
              <p><b>Name:</b> {profile.username}</p>
              <p><b>Email:</b> {profile.email}</p>
              <p><b>Age:</b> {profile.age || 'Not specified'}</p>
              <p><b>Pincode:</b> {profile.pincode || 'Not specified'}</p>
            </>
          ) : (
            <p className="text-warning">No profile data.</p>
          )}
          <button className="btn btn-secondary mt-3 w-100" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
