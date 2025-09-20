import { useState, useEffect, type FC } from "react"
import { Dropdown, Navbar, Container, Nav } from "react-bootstrap"
import Profile from "./Profile"
import UpdatePreferences from "./UpdatePreferences"
import { FaUser, FaHeart, FaSignOutAlt } from "react-icons/fa"

interface HomepageProps {
  token: string
  onLogout: () => void
}

const Homepage: FC<HomepageProps> = ({ token, onLogout }) => {
  const [preferences, setPreferences] = useState<string[]>([])
  const [view, setView] = useState<"none" | "profile" | "preferences">("none")
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)

  // Fetch preferences when opening preferences view
  useEffect(() => {
    if (view !== "preferences") return
    const fetchPreferences = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch("http://localhost:8000/user/preferences", {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json()
          setPreferences(data.genres || [])
        } else if (res.status === 404) {
          setPreferences([])
        } else {
          setError("Failed to load preferences")
        }
      } catch {
        setError("Network error while loading preferences")
      } finally {
        setLoading(false)
      }
    }
    fetchPreferences()
  }, [view, token])

  // Save updated preferences
  const savePreferences = async (genres: string[]) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("http://localhost:8000/user/preferences", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ genres }),
      })
      if (res.ok) {
        setPreferences(genres)
        setView("none")
      } else {
        const data = await res.json()
        setError(data.detail || "Failed to save preferences")
      }
    } catch {
      setError("Network error while saving preferences")
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Navbar bg="light" expand="lg" fixed="top" className="shadow-sm">
        <Container>
          <Navbar.Brand href="#">BibliophileAI</Navbar.Brand>
          <Nav className="ms-auto">
            <Dropdown>
              <Dropdown.Toggle variant="secondary" id="dropdown-menu">
                Menu
              </Dropdown.Toggle>
              <Dropdown.Menu align="end">
                <Dropdown.Item onClick={() => setView("profile")}>
                  <FaUser className="me-2" /> Profile
                </Dropdown.Item>
                <Dropdown.Item onClick={() => setView("preferences")}>
                  <FaHeart className="me-2" /> Preferences
                </Dropdown.Item>
                <Dropdown.Item
                  onClick={() => {
                    onLogout()
                    window.location.href = "/"
                  }}
                  className="text-danger"
                >
                  <FaSignOutAlt className="me-2" /> Logout
                </Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>
          </Nav>
        </Container>
      </Navbar>

      <div className="container" style={{ marginTop: "75px" }}>
        {view === "profile" && (
          <Profile token={token} onClose={() => setView("none")} />
        )}
        {view === "preferences" && (
          <UpdatePreferences
            initialGenres={preferences}
            loading={loading}
            onSave={savePreferences}
            onCancel={() => setView("none")}
          />
        )}
        {/* No content if view === "none" */}
      </div>
    </>
  )
}

export default Homepage
