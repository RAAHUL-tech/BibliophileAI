import { useState, useEffect, useRef, useMemo, type FC } from "react"
import { Dropdown, Navbar, Container, Nav, Spinner, Card, Button } from "react-bootstrap"
import Profile from "./Profile"
import UpdateGenrePreferences from "./UpdateGenrePreferences"
import UpdateAuthorPreferences from "./UpdateAuthorPreferences"
import BookOverlay from "./BookOverlay"
import { useTheme } from "../context/ThemeContext"
import { FaUser, FaHeart, FaSignOutAlt, FaUserPlus, FaChevronLeft, FaChevronRight, FaSearch, FaSun, FaMoon, FaTimes } from "react-icons/fa"
import "./NetflixStyles.css"
import "./SharedStyles.css"

interface BookRecommendation {
  id: string
  title: string
  authors: string[]
  categories: string[]
  thumbnail_url?: string
  download_link?: string
  score?: number
}

interface RecommendationCategory {
  category: string
  description: string
  books: BookRecommendation[]
}

interface UserPreferences {
  id: string | null
  user_id: string
  genres: string[]
  authors: string[]
}

interface UserSuggestion {
  id: string
  username: string
}

interface HomepageProps {
  token: string
  onLogout: () => void
}

/** Fixed row order for recommendation categories. Only rows present in this list are shown, in this order. */
const CATEGORY_ORDER: string[] = [
  "Continue Reading",
  "Top Picks",
  "Content-Based Recommendations",
  "Collaborative Filtering",
  "Social Recommendations",
  "Session-Based",
  "Trending Now",
  "For You (LinUCB)",
]

/** Order categories by CATEGORY_ORDER and skip categories with no books. Stable across reload/login. */
function orderCategories(categories: RecommendationCategory[]): RecommendationCategory[] {
  const byName = new Map<string, RecommendationCategory>()
  for (const c of categories) {
    byName.set(c.category, c)
  }
  const ordered: RecommendationCategory[] = []
  for (const name of CATEGORY_ORDER) {
    const cat = byName.get(name)
    if (cat && cat.books && cat.books.length > 0) {
      ordered.push(cat)
    }
  }
  return ordered
}

/** Map backend category names to engaging, user-facing titles */
function getDisplayTitle(backendCategory: string): string {
  const map: Record<string, string> = {
    "Content-Based Recommendations": "Because you liked...",
    "Collaborative Filtering": "Readers like you also loved",
    "Social Recommendations": "From your community",
    "Session-Based": "Based on your recent reads",
    "Trending Now": "Trending now",
    "For You (LinUCB)": "Just for you",
    "Continue Reading": "Continue reading",
    "Top Picks": "Top picks for you",
  }
  if (map[backendCategory]) return map[backendCategory]
  if (backendCategory.startsWith("More in ")) return backendCategory
  return backendCategory
}

const Homepage: FC<HomepageProps> = ({ token, onLogout }) => {
  const { theme, toggleTheme } = useTheme()
  const [preferences, setPreferences] = useState<UserPreferences | null>(null)
  const [view, setView] = useState<"none" | "profile" | "preferences" | "author_preferences">("none")
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string | null>(null)
  const [recommendations, setRecommendations] = useState<BookRecommendation[]>([])
  const [recommendationCategories, setRecommendationCategories] = useState<RecommendationCategory[]>([])
  const [loadingRecs, setLoadingRecs] = useState<boolean>(true)
  const [recsError, setRecsError] = useState<string | null>(null)
  const [selectedBook, setSelectedBook] = useState<BookRecommendation | null>(null)
  const [searchExpanded, setSearchExpanded] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")

  // Follower suggestions state
  const [followerSuggestions, setFollowerSuggestions] = useState<UserSuggestion[]>([])
  const [loadingSuggestions, setLoadingSuggestions] = useState<boolean>(true)
  const [followingInProgress, setFollowingInProgress] = useState<Set<string>>(new Set())

  // Client-side search filter (no backend calls)
  const filteredCategories = useMemo(() => {
    if (!searchQuery.trim()) return recommendationCategories
    const q = searchQuery.trim().toLowerCase()
    return recommendationCategories
      .map((cat) => ({
        ...cat,
        books: cat.books.filter(
          (b) =>
            b.title?.toLowerCase().includes(q) ||
            b.authors?.some((a) => String(a).toLowerCase().includes(q)) ||
            b.categories?.some((c) => String(c).toLowerCase().includes(q))
        ),
      }))
      .filter((cat) => cat.books.length > 0)
  }, [recommendationCategories, searchQuery])

  // Fetch recommendations on first load
  useEffect(() => {
    setLoadingRecs(true)
    setRecsError(null)
    const fetchRecs = async () => {
      try {
        const res = await fetch("http://localhost:8080/api/v1/recommend/combined", {
          headers: { 
            Authorization: `Bearer ${token}` ,
            'Content-Type': 'application/json',
            'User-Agent': navigator.userAgent
          }
        })
        if (res.ok) {
          const data = await res.json()
          // Handle new category-based format: enforce fixed row order and skip empty categories
          if (data.categories) {
            const ordered = orderCategories(data.categories || [])
            setRecommendationCategories(ordered)
            // Flatten for backward compatibility
            const allBooks = ordered.flatMap((cat: RecommendationCategory) => cat.books)
            setRecommendations(allBooks)
          } else {
            // Fallback to old format
            setRecommendations(data.recommendations || [])
            setRecommendationCategories([])
          }
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

  // Fetch follower suggestions
  useEffect(() => {
    const fetchSuggestions = async () => {
      setLoadingSuggestions(true)
      try {
        const res = await fetch("http://localhost:8080/api/v1/user/follower-suggestions?limit=10", {
          headers: { Authorization: `Bearer ${token}` }
        })
        if (res.ok) {
          const data = await res.json()
          setFollowerSuggestions(data.suggestions || [])
        } else {
          console.error("Failed to load follower suggestions")
        }
      } catch (err) {
        console.error("Network error loading suggestions:", err)
      } finally {
        setLoadingSuggestions(false)
      }
    }
    fetchSuggestions()
  }, [token])

  // Fetch preferences when opening preferences view
  useEffect(() => {
    if (view !== "preferences" && view !== "author_preferences") return
    const fetchPreferences = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch("http://localhost:8080/api/v1/user/preferences", {
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

  // Handle follow user
  const handleFollowUser = async (userId: string) => {
    setFollowingInProgress(prev => new Set(prev).add(userId))
    try {
      const res = await fetch(`http://localhost:8080/api/v1/user/follow/${userId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` }
      })
      
      if (res.ok) {
        // Remove from suggestions after successful follow
        setFollowerSuggestions(prev => prev.filter(u => u.id !== userId))
      } else {
        alert("Failed to follow user")
      }
    } catch (err) {
      console.error("Error following user:", err)
      alert("Network error while following user")
    } finally {
      setFollowingInProgress(prev => {
        const newSet = new Set(prev)
        newSet.delete(userId)
        return newSet
      })
    }
  }

  // Save updated genre preferences
  const saveGenrePreferences = async (genres: string[]) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch("http://localhost:8080/api/v1/user/preferences", {
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
      const res = await fetch("http://localhost:8080/api/v1/user/preferences", {
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
      <Navbar bg="dark" variant="dark" expand="lg" fixed="top" className="shadow-sm bib-nav">
        <Container fluid className="d-flex align-items-center">
          <Navbar.Brand href="#" className="me-3">BibliophileAI</Navbar.Brand>
          {searchExpanded && (
            <div className="bib-search-expanded">
              <input
                type="search"
                placeholder="Search titles, authors..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onBlur={() => { if (!searchQuery.trim()) setSearchExpanded(false); }}
                autoFocus
                aria-label="Search books"
              />
            </div>
          )}
          <Nav className="ms-auto d-flex align-items-center">
            <button
              type="button"
              className="bib-search-trigger"
              onClick={() => setSearchExpanded(true)}
              aria-label="Search"
            >
              <FaSearch />
            </button>
            <button
              type="button"
              className="bib-theme-toggle"
              onClick={toggleTheme}
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            >
              {theme === "dark" ? <FaSun /> : <FaMoon />}
            </button>
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

      <div
        className={`container-fluid ${searchExpanded ? "bib-blurred" : ""}`}
        style={{ marginTop: "75px", backgroundColor: "var(--bib-bg)", minHeight: "100vh", padding: 0, transition: "filter 0.3s ease" }}
      >
        <div className="row" style={{ margin: 0 }}>
          {/* Main Content Area */}
          <div className="col-lg-9 col-md-8" style={{ padding: 0 }}>
            {error && (view === "preferences" || view === "author_preferences") && (
              <div className="bib-alert-danger m-3 d-flex align-items-center justify-content-between">
                <span>{error}</span>
                <button type="button" className="bib-btn-secondary" style={{ padding: "0.25rem 0.5rem", fontSize: "0.875rem" }} onClick={() => setError(null)} aria-label="Dismiss">
                  <FaTimes />
                </button>
              </div>
            )}
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
                  <div className="netflix-loading-state">
                    <div className="netflix-loading-spinner" role="status" aria-label="Loading" />
                    <span className="netflix-loading-text">Loading recommendations for you...</span>
                  </div>
                ) : recsError ? (
                  <div className="bib-alert-danger m-3">{recsError}</div>
                ) : filteredCategories.length > 0 ? (
                  <div className="netflix-container">
                    {filteredCategories.map((category) => (
                      <NetflixRow
                        key={category.category}
                        category={category}
                        displayTitle={getDisplayTitle(category.category)}
                        onBookClick={setSelectedBook}
                      />
                    ))}
                  </div>
                ) : recommendationCategories.length > 0 ? (
                  <div className="netflix-container py-5 px-4 text-center" style={{ color: "var(--bib-text-muted)" }}>
                    <p className="mb-0">No books match your search. Try a different term or clear the search.</p>
                  </div>
                ) : (
                  <div className="netflix-container py-5 px-4">
                    <h2 className="text-center" style={{ color: "var(--bib-text-title)" }}>Recommended for you</h2>
                    <div className="row g-3">
                      {recommendations.map((book) => (
                        <div key={book.id} className="col-md-6 col-lg-4">
                          <div
                            className="netflix-item"
                            style={{ cursor: "pointer" }}
                            onClick={() => setSelectedBook(book)}
                          >
                            {book.thumbnail_url ? (
                              <img src={book.thumbnail_url} className="netflix-item-image" alt={book.title} />
                            ) : (
                              <div className="netflix-item-placeholder">
                                <span>{book.title}</span>
                              </div>
                            )}
                            <div className="netflix-item-overlay">
                              <h4>{book.title}</h4>
                              {book.authors?.length > 0 && <p>{book.authors.join(", ")}</p>}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}
          </div>

          {/* Right Sidebar - Follower Suggestions */}
          <div className="col-lg-3 col-md-4" style={{ padding: "1rem" }}>
            <div className="sticky-top" style={{ top: "85px" }}>
              <Card className="shadow-sm" style={{ backgroundColor: "var(--bib-bg-elevated)", borderColor: "var(--bib-border)" }}>
                <Card.Header style={{ backgroundColor: "var(--bib-card-bg)", borderColor: "var(--bib-border)", color: "var(--bib-text)" }}>
                  <h5 className="mb-0">👥 Suggested Users</h5>
                </Card.Header>
                <Card.Body style={{ maxHeight: "70vh", overflowY: "auto", backgroundColor: "var(--bib-bg-elevated)", color: "var(--bib-text)" }}>
                  {loadingSuggestions ? (
                    <div className="text-center py-3">
                      <Spinner animation="border" size="sm" />
                      <p className="mt-2 text-muted small">Loading...</p>
                    </div>
                  ) : followerSuggestions.length === 0 ? (
                    <p className="text-muted text-center">No suggestions available</p>
                  ) : (
                    <div className="d-flex flex-column gap-3">
                      {followerSuggestions.map((user) => (
                        <div
                          key={user.id}
                          className="d-flex justify-content-between align-items-center p-2 border rounded"
                        >
                          <span className="fw-medium">{user.username}</span>
                          <Button
                            size="sm"
                            variant="outline-primary"
                            onClick={() => handleFollowUser(user.id)}
                            disabled={followingInProgress.has(user.id)}
                          >
                            {followingInProgress.has(user.id) ? (
                              <Spinner animation="border" size="sm" />
                            ) : (
                              <>
                                <FaUserPlus className="me-1" />
                                Follow
                              </>
                            )}
                          </Button>
                        </div>
                      ))}
                    </div>
                  )}
                </Card.Body>
              </Card>
            </div>
          </div>
        </div>
      </div>

      {selectedBook && (
        <BookOverlay
          book={selectedBook}
          token={token}
          onClose={() => setSelectedBook(null)}
        />
      )}
    </>
  )
}

// Netflix-style Row Component with horizontal scrolling
interface NetflixRowProps {
  category: RecommendationCategory
  displayTitle?: string
  onBookClick: (book: BookRecommendation) => void
}

const NetflixRow: FC<NetflixRowProps> = ({ category, displayTitle, onBookClick }) => {
  const sliderRef = useRef<HTMLDivElement>(null)
  const title = displayTitle ?? category.category

  const scrollLeft = () => {
    if (sliderRef.current) {
      sliderRef.current.scrollBy({ left: -600, behavior: 'smooth' })
    }
  }

  const scrollRight = () => {
    if (sliderRef.current) {
      sliderRef.current.scrollBy({ left: 600, behavior: 'smooth' })
    }
  }

  return (
    <div className="netflix-row">
      <h3 className="netflix-row-title">{title}</h3>
      <p className="netflix-row-description">{category.description}</p>
      <div className="netflix-row-content" style={{ position: 'relative' }}>
        <button 
          className="netflix-scroll-button left"
          onClick={scrollLeft}
          aria-label="Scroll left"
        >
          <FaChevronLeft />
        </button>
        <div className="netflix-slider" ref={sliderRef}>
          {category.books.map((book) => (
            <div
              key={book.id}
              className="netflix-item"
              onClick={() => onBookClick(book)}
            >
              {book.thumbnail_url ? (
                <img 
                  src={book.thumbnail_url} 
                  alt={book.title}
                  className="netflix-item-image"
                />
              ) : (
                <div className="netflix-item-placeholder">
                  <span>{book.title}</span>
                </div>
              )}
              <div className="netflix-item-overlay">
                <h4>{book.title}</h4>
                {book.authors && book.authors.length > 0 && (
                  <p>{book.authors.join(", ")}</p>
                )}
              </div>
            </div>
          ))}
        </div>
        <button 
          className="netflix-scroll-button right"
          onClick={scrollRight}
          aria-label="Scroll right"
        >
          <FaChevronRight />
        </button>
      </div>
    </div>
  )
}

export default Homepage
