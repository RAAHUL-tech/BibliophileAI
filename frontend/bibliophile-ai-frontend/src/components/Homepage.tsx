import { useState, useEffect, type FC } from "react"
import { Dropdown, Navbar, Container, Nav, Spinner, Alert } from "react-bootstrap"
import Profile from "./Profile"
import UpdateGenrePreferences from "./UpdateGenrePreferences"
import UpdateAuthorPreferences from "./UpdateAuthorPreferences"
import BookView from "./BookView";
import { FaUser, FaHeart, FaSignOutAlt } from "react-icons/fa"


interface BookRecommendation {
  id: string
  title: string
  authors: string[]
  categories: string[]
  thumbnail_url?: string
  download_link?: string
}

interface UserPreferences {
  id: string | null
  user_id: string
  genres: string[]
  authors: string[]
}

interface HomepageProps {
  token: string
  onLogout: () => void
}

const Homepage: FC<HomepageProps> = ({ token, onLogout }) => {
  const [preferences, setPreferences] = useState<UserPreferences | null>(null)
  const [view, setView] = useState<"none" | "profile" | "preferences" | "author_preferences">("none")
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [recommendations, setRecommendations] = useState<BookRecommendation[]>([])
  const [loadingRecs, setLoadingRecs] = useState<boolean>(true)
  const [recsError, setRecsError] = useState<string | null>(null)
  const [selectedBook, setSelectedBook] = useState<BookRecommendation | null>(null);

  // Fetch recommendations on first load
  useEffect(() => {
    setLoadingRecs(true)
    setRecsError(null)
    const fetchRecs = async () => {
      try {
        const res = await fetch("http://localhost:8001/api/v1/recommend/combined", {
          headers: { Authorization: `Bearer ${token}` }
        })
        if (res.ok) {
          const data = await res.json()
          setRecommendations(data.recommendations || [])
        } else {
          setRecsError("Failed to load recommendations")
        }
      } catch {
        setRecsError("Network error while loading recommendations")
      } finally {
        setLoadingRecs(false)
      }
    }
    fetchRecs()
  }, [token])

  // Fetch preferences when opening preferences view
  useEffect(() => {
    if (view !== "preferences" && view !== "author_preferences") return
    const fetchPreferences = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch("http://localhost:8000/api/v1/user/preferences", {
          headers: { Authorization: `Bearer ${token}` },
        })
        if (res.ok) {
          const data = await res.json()
          setPreferences(data)
        } else if (res.status === 404) {
          setPreferences({
            id: null,
            user_id: "",
            genres: [],
            authors: []
          })
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

  // Save updated genre preferences
  const saveGenrePreferences = async (genres: string[]) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("http://localhost:8000/api/v1/user/preferences", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ 
          genres,
          authors: preferences?.authors || []
        }),
      })
      if (res.ok) {
        setPreferences(prev => prev ? { ...prev, genres } : null)
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

  // Save updated author preferences
  const saveAuthorPreferences = async (authors: string[]) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("http://localhost:8000/api/v1/user/preferences", {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ 
          genres: preferences?.genres || [],
          authors 
        }),
      })
      if (res.ok) {
        setPreferences(prev => prev ? { ...prev, authors } : null)
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
                  <FaHeart className="me-2" /> Genre Preferences
                </Dropdown.Item>
                <Dropdown.Item onClick={() => setView("author_preferences")}>
                  <FaHeart className="me-2" /> Author Preferences
                </Dropdown.Item>
                <Dropdown.Item
                  onClick={() => {
                    onLogout()
                    
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
  {selectedBook ? (
    <BookView book={selectedBook} token={token} onBack={() => setSelectedBook(null)} />
  ) : (
    <>
      {view === "profile" && (
        <Profile token={token} onClose={() => setView("none")} />
      )}
      {view === "preferences" && preferences && (
        <UpdateGenrePreferences
          initialGenres={preferences.genres}
          loading={loading}
          onSave={saveGenrePreferences}
          onCancel={() => setView("none")}
        />
      )}
      {view === "author_preferences" && preferences && (
        <UpdateAuthorPreferences
          initialAuthors={preferences.authors}
          loading={loading}
          onSave={saveAuthorPreferences}
          onCancel={() => setView("none")}
          token={token}
        />
      )}
      {view === "none" && (
        <>
          {loadingRecs ? (
            <div className="d-flex justify-content-center align-items-center" style={{ minHeight: "60vh" }}>
              <Spinner animation="border" role="status" />
              <span className="ms-3">Loading recommendations...</span>
            </div>
          ) : recsError ? (
            <Alert variant="danger">{recsError}</Alert>
          ) : (
            <div>
              <h2>Recommended for you</h2>
              <div className="row">
                {recommendations.map((book) => (
                  <div key={book.id} className="col-md-4 col-lg-3 mb-4">
                    <div
                      className="card h-100"
                      style={{ cursor: "pointer" }}
                      onClick={() => setSelectedBook(book)}
                    >
                      {book.thumbnail_url && (
                        <img src={book.thumbnail_url} className="card-img-top" alt={book.title} />
                      )}
                      <div className="card-body">
                        <h5 className="card-title">{book.title}</h5>
                        <p className="card-text">
                          {book.authors && book.authors.join(", ")}
                        </p>
                        <p className="card-text">
                          {book.categories && book.categories.join(", ")}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </>
  )}
</div>
    </>
  )
}

export default Homepage
