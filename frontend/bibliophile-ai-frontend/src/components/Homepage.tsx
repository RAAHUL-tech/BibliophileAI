import { useState, useEffect, useRef, useMemo, useCallback, type FC } from "react"
import { Dropdown, Navbar, Container, Nav } from "react-bootstrap"
import Profile from "./Profile"
import UpdateGenrePreferences from "./UpdateGenrePreferences"
import UpdateAuthorPreferences from "./UpdateAuthorPreferences"
import BookOverlay from "./BookOverlay"
import SearchOverlay from "./SearchOverlay"
import { useTheme } from "../context/ThemeContext"
import { FaUser, FaHeart, FaSignOutAlt, FaUserPlus, FaChevronLeft, FaChevronRight, FaSearch, FaSun, FaMoon, FaTimes, FaSpinner } from "react-icons/fa"
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
  const [searchResults, setSearchResults] = useState<BookRecommendation[]>([])
  const [searchLoading, setSearchLoading] = useState(false)
  const [searchError, setSearchError] = useState<string | null>(null)
  const [searchOverlayOpen, setSearchOverlayOpen] = useState(false)
  const [navScrolled, setNavScrolled] = useState(false)

  // Follower suggestions state
  const [followerSuggestions, setFollowerSuggestions] = useState<UserSuggestion[]>([])
  const [loadingSuggestions, setLoadingSuggestions] = useState<boolean>(true)
  const [followingInProgress, setFollowingInProgress] = useState<Set<string>>(new Set())

  // Navbar scroll effect
  useEffect(() => {
    const onScroll = () => setNavScrolled(window.scrollY > 20)
    window.addEventListener("scroll", onScroll, { passive: true })
    return () => window.removeEventListener("scroll", onScroll)
  }, [])

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

  const handleSearchSubmit = async () => {
    const query = searchQuery.trim()
    if (!query) return
    setSearchLoading(true)
    setSearchError(null)
    try {
      const res = await fetch("http://localhost:8080/api/v1/search/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ query }),
      })
      if (!res.ok) {
        throw new Error("Search request failed")
      }
      const data = await res.json()
      setSearchResults(data.results || [])
      setSearchOverlayOpen(true)
    } catch (err) {
      console.error("Search error", err)
      setSearchError("Failed to search books. Please try again.")
      setSearchResults([])
      setSearchOverlayOpen(true)
    } finally {
      setSearchLoading(false)
    }
  }

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
      <Navbar
        bg={theme === "dark" ? "dark" : "light"}
        variant={theme === "dark" ? "dark" : "light"}
        expand="lg"
        fixed="top"
        className={`shadow-sm bib-nav${navScrolled ? " bib-nav-scrolled" : ""}`}
      >
        <Container fluid className="d-flex align-items-center">
          <Navbar.Brand href="#" className="me-3">📚 BibliophileAI</Navbar.Brand>
          {searchExpanded && (
            <div className="bib-search-expanded">
              <input
                type="search"
                placeholder="Search titles, authors..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault()
                    void handleSearchSubmit()
                  }
                }}
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
                  onClick={() => { onLogout() }}
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
        style={{
          marginTop: "75px",
          backgroundColor: "var(--bib-bg)",
          minHeight: "100vh",
          padding: 0,
          transition: "filter 0.3s ease, background-color 0.35s ease"
        }}
      >
        <div className="row" style={{ margin: 0 }}>
          {/* Main Content Area */}
          <div className="col-lg-9 col-md-8" style={{ padding: 0 }}>
            {error && (view === "preferences" || view === "author_preferences") && (
              <div className="bib-alert-danger m-3 d-flex align-items-center justify-content-between">
                <span>{error}</span>
                <button
                  type="button"
                  className="bib-btn-secondary"
                  style={{ padding: "0.25rem 0.5rem", fontSize: "0.875rem" }}
                  onClick={() => setError(null)}
                  aria-label="Dismiss"
                >
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

          {/* Right Sidebar — Follower Suggestions */}
          <div className="col-lg-3 col-md-4" style={{ padding: "1rem" }}>
            <div className="sticky-top" style={{ top: "85px" }}>
              <div
                className="rounded-3 overflow-hidden"
                style={{
                  background: "var(--bib-bg-elevated)",
                  border: "1px solid var(--bib-border)",
                  boxShadow: "var(--bib-shadow-md)",
                  transition: "background 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease"
                }}
              >
                {/* Sidebar header */}
                <div
                  style={{
                    background: "var(--bib-card-header)",
                    borderBottom: "1px solid var(--bib-border)",
                    padding: "0.85rem 1rem",
                    transition: "background 0.35s ease"
                  }}
                >
                  <h5
                    className="mb-0"
                    style={{
                      color: "var(--bib-text-title)",
                      fontWeight: 700,
                      fontSize: "1rem",
                      letterSpacing: "-0.01em"
                    }}
                  >
                    👥 Suggested Users
                  </h5>
                </div>

                {/* Sidebar body */}
                <div
                  style={{
                    maxHeight: "70vh",
                    overflowY: "auto",
                    padding: "0.75rem",
                    background: "var(--bib-bg-elevated)",
                    color: "var(--bib-text)",
                    transition: "background 0.35s ease"
                  }}
                >
                  {loadingSuggestions ? (
                    <div className="text-center py-4">
                      <FaSpinner
                        style={{
                          fontSize: "1.5rem",
                          animation: "spin 1s linear infinite",
                          color: "var(--bib-accent)"
                        }}
                      />
                      <p className="mt-2 small mb-0" style={{ color: "var(--bib-text-muted)" }}>
                        Loading...
                      </p>
                    </div>
                  ) : followerSuggestions.length === 0 ? (
                    <p
                      className="text-center py-3 mb-0 small"
                      style={{ color: "var(--bib-text-muted)" }}
                    >
                      No suggestions available
                    </p>
                  ) : (
                    <div className="d-flex flex-column gap-2">
                      {followerSuggestions.map((user, idx) => (
                        <div
                          key={user.id}
                          className="bib-suggestion-card"
                          style={{ animationDelay: `${idx * 0.06}s` }}
                        >
                          <div className="d-flex align-items-center gap-2">
                            <div
                              style={{
                                width: 32,
                                height: 32,
                                borderRadius: "50%",
                                background: "linear-gradient(135deg, #667eea, #764ba2)",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                fontSize: "0.85rem",
                                fontWeight: 700,
                                color: "white",
                                flexShrink: 0,
                                textTransform: "uppercase"
                              }}
                            >
                              {user.username.charAt(0).toUpperCase()}
                            </div>
                            <span
                              style={{
                                color: "var(--bib-text)",
                                fontWeight: 600,
                                fontSize: "0.88rem"
                              }}
                            >
                              {user.username}
                            </span>
                          </div>
                          <button
                            className="bib-btn-primary"
                            style={{
                              padding: "0.3rem 0.75rem",
                              fontSize: "0.8rem",
                              borderRadius: "20px",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "0.3rem"
                            }}
                            onClick={() => handleFollowUser(user.id)}
                            disabled={followingInProgress.has(user.id)}
                          >
                            {followingInProgress.has(user.id) ? (
                              <FaSpinner style={{ animation: "spin 1s linear infinite" }} />
                            ) : (
                              <>
                                <FaUserPlus style={{ fontSize: "0.75rem" }} />
                                Follow
                              </>
                            )}
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
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
      {searchOverlayOpen && (
        <SearchOverlay
          results={searchResults}
          loading={searchLoading}
          error={searchError}
          onClose={() => {
            setSearchOverlayOpen(false)
            setSearchError(null)
          }}
          onBookClick={(book) => {
            setSelectedBook(book)
            setSearchOverlayOpen(false)
          }}
        />
      )}
    </>
  )
}

// Netflix-style Row Component with horizontal scrolling + touch swipe support
interface NetflixRowProps {
  category: RecommendationCategory
  displayTitle?: string
  onBookClick: (book: BookRecommendation) => void
}

const NetflixRow: FC<NetflixRowProps> = ({ category, displayTitle, onBookClick }) => {
  const sliderRef = useRef<HTMLDivElement>(null)
  const touchStartX = useRef(0)
  const touchEndX = useRef(0)
  const title = displayTitle ?? category.category

  const scrollLeft = useCallback(() => {
    sliderRef.current?.scrollBy({ left: -600, behavior: "smooth" })
  }, [])

  const scrollRight = useCallback(() => {
    sliderRef.current?.scrollBy({ left: 600, behavior: "smooth" })
  }, [])

  // Touch swipe: trigger scroll after swipe of > 60px
  const onTouchStart = useCallback((e: React.TouchEvent) => {
    touchStartX.current = e.touches[0].clientX
    touchEndX.current = e.touches[0].clientX
  }, [])

  const onTouchMove = useCallback((e: React.TouchEvent) => {
    touchEndX.current = e.touches[0].clientX
  }, [])

  const onTouchEnd = useCallback(() => {
    const delta = touchStartX.current - touchEndX.current
    if (Math.abs(delta) > 60) {
      if (delta > 0) scrollRight()
      else scrollLeft()
    }
  }, [scrollLeft, scrollRight])

  return (
    <div className="netflix-row">
      <h3 className="netflix-row-title">{title}</h3>
      {category.description && (
        <p className="netflix-row-description">{category.description}</p>
      )}
      <div className="netflix-row-content" style={{ position: 'relative' }}>
        <button
          className="netflix-scroll-button left"
          onClick={scrollLeft}
          aria-label="Scroll left"
        >
          <FaChevronLeft />
        </button>
        <div
          className="netflix-slider"
          ref={sliderRef}
          onTouchStart={onTouchStart}
          onTouchMove={onTouchMove}
          onTouchEnd={onTouchEnd}
        >
          {category.books.map((book) => (
            <div
              key={book.id}
              className="netflix-item"
              onClick={() => onBookClick(book)}
              role="button"
              tabIndex={0}
              onKeyDown={(e) => e.key === "Enter" && onBookClick(book)}
              aria-label={`Open ${book.title}`}
            >
              {book.thumbnail_url ? (
                <img
                  src={book.thumbnail_url}
                  alt={book.title}
                  className="netflix-item-image"
                  loading="lazy"
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
