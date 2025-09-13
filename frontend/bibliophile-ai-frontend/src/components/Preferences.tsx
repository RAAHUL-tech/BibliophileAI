import { useState } from "react"

const allGenres = [
  "Fantasy",
  "Romance",
  "Science Fiction",
  "Mystery",
  "Thriller",
  "Non-fiction",
  "Historical",
  "Young Adult",
  "Horror",
  "Biography",
]

// Genre emoji mapping for visual appeal
const genreEmojis: { [key: string]: string } = {
  "Fantasy": "🧙‍♂️",
  "Romance": "💕",
  "Science Fiction": "🚀",
  "Mystery": "🔍",
  "Thriller": "⚡",
  "Non-fiction": "📚",
  "Historical": "🏛️",
  "Young Adult": "🌟",
  "Horror": "👻",
  "Biography": "👤",
}

type PreferencesProps = {
  initialSelectedGenres?: string[]
  onSave: (genres: string[]) => void
  loading?: boolean
}

export default function Preferences({
  initialSelectedGenres = [],
  onSave,
  loading = false,
}: PreferencesProps) {
  const [selectedGenres, setSelectedGenres] = useState<string[]>(initialSelectedGenres)

  const toggleGenre = (genre: string) => {
    setSelectedGenres((genres) =>
      genres.includes(genre)
        ? genres.filter((g) => g !== genre)
        : [...genres, genre]
    )
  }

  return (
    <div
      className="min-vh-100 d-flex align-items-center justify-content-center position-relative"
      style={{
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%)',
        padding: '2rem 0'
      }}
    >
      {/* Animated Background Pattern */}
      <div
        className="position-absolute w-100 h-100 opacity-10"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.4'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3Ccircle cx='15' cy='15' r='2'/%3E%3Ccircle cx='45' cy='45' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          animation: 'float 20s ease-in-out infinite',
          pointerEvents: "none"
        }}
      />
      {/* Floating Decorative Elements */}
      <div
        className="position-absolute"
        style={{
          top: '10%', left: '5%', animation: 'bounce 4s infinite', pointerEvents: "none"
        }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-3" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '2rem' }}>📖</span>
        </div>
      </div>
      <div
        className="position-absolute"
        style={{
          top: '20%', right: '8%', animation: 'bounce 4s infinite 1s', pointerEvents: "none"
        }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-2" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '1.5rem' }}>✨</span>
        </div>
      </div>
      <div
        className="position-absolute"
        style={{
          bottom: '15%', left: '8%', animation: 'bounce 4s infinite 2s', pointerEvents: "none"
        }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-2" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '1.5rem' }}>🎭</span>
        </div>
      </div>

      <div className="container">
        <div className="row justify-content-center">
          <div className="col-lg-8 col-xl-7">
            <div
              className="card shadow-lg border-0 position-relative overflow-hidden"
              style={{
                background: 'rgba(255, 255, 255, 0.95)',
                backdropFilter: 'blur(20px)',
                borderRadius: '2rem',
                border: '1px solid rgba(255, 255, 255, 0.2)'
              }}
            >
              {/* Card Header */}
              <div
                className="card-header border-0 text-center py-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%)',
                  borderRadius: '2rem 2rem 0 0'
                }}
              >
                <div className="mb-3">
                  <span style={{ fontSize: '3rem', filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))' }}>
                    📚✨
                  </span>
                </div>
                <h2
                  className="display-6 fw-bold mb-2"
                  style={{
                    background: 'linear-gradient(135deg, #667eea, #764ba2)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text'
                  }}
                >
                  Discover Your Reading Universe
                </h2>
                <p className="lead text-muted mb-0">
                  Select your favorite genres to get personalized book recommendations
                </p>
                {selectedGenres.length > 0 && (
                  <div className="mt-3">
                    <span
                      className="badge px-3 py-2 rounded-pill"
                      style={{
                        background: 'linear-gradient(135deg, #667eea, #764ba2)',
                        color: 'white',
                        fontSize: '0.9rem',
                        animation: 'pulse 2s infinite'
                      }}
                    >
                      {selectedGenres.length} {selectedGenres.length === 1 ? 'genre' : 'genres'} selected
                    </span>
                  </div>
                )}
              </div>
              {/* Card Body with Genre Selection */}
              <div className="card-body p-4">
                <div className="row g-3">
                  {allGenres.map((genre, index) => (
                    <div key={genre} className="col-6 col-md-4 col-lg-3">
                      <button
                        type="button"
                        className={`btn w-100 h-100 border-2 position-relative overflow-hidden ${
                          selectedGenres.includes(genre)
                            ? "btn-primary shadow-lg"
                            : "btn-outline-primary"
                        }`}
                        onClick={() => toggleGenre(genre)}
                        style={{
                          minHeight: '80px',
                          borderRadius: '1rem',
                          transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                          transform: selectedGenres.includes(genre) ? 'translateY(-2px)' : 'none',
                          boxShadow: selectedGenres.includes(genre)
                            ? '0 10px 30px rgba(102, 126, 234, 0.3)'
                            : '0 4px 15px rgba(0, 0, 0, 0.1)',
                          animationDelay: `${index * 0.1}s`,
                          background: selectedGenres.includes(genre)
                            ? 'linear-gradient(135deg, #667eea, #764ba2)'
                            : 'transparent',
                          border: selectedGenres.includes(genre)
                            ? '2px solid transparent'
                            : '2px solid #667eea'
                        }}
                      >
                        {selectedGenres.includes(genre) && (
                          <div
                            className="position-absolute top-0 start-0 w-100 h-100"
                            style={{
                              background: 'radial-gradient(circle, rgba(255,255,255,0.3) 0%, transparent 70%)',
                              borderRadius: '1rem',
                              animation: 'ripple 1.5s ease-out infinite',
                              pointerEvents: "none"
                            }}
                          />
                        )}
                        <div className="d-flex flex-column align-items-center justify-content-center h-100">
                          <span
                            className="fs-4 mb-1"
                            style={{
                              filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))',
                              transform: selectedGenres.includes(genre) ? 'scale(1.1)' : 'scale(1)',
                              transition: 'transform 0.3s ease'
                            }}
                          >
                            {genreEmojis[genre]}
                          </span>
                          <span
                            className={`fw-bold small text-center ${
                              selectedGenres.includes(genre) ? 'text-white' : 'text-primary'
                            }`}
                            style={{
                              fontSize: '0.8rem',
                              lineHeight: '1.2',
                              textShadow: selectedGenres.includes(genre) ? '0 1px 2px rgba(0,0,0,0.2)' : 'none'
                            }}
                          >
                            {genre}
                          </span>
                        </div>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
              {/* Card Footer with Action Button */}
              <div
                className="card-footer border-0 text-center py-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%)',
                  borderRadius: '0 0 2rem 2rem'
                }}
              >
                <button
                  className={`btn btn-lg px-5 py-3 rounded-pill fw-bold position-relative overflow-hidden ${
                    selectedGenres.length === 0 ? 'btn-outline-secondary' : 'btn-success'
                  }`}
                  disabled={selectedGenres.length === 0 || loading}
                  onClick={() => onSave(selectedGenres)}
                  style={{
                    minWidth: '200px',
                    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                    boxShadow: selectedGenres.length > 0 && !loading
                      ? '0 8px 25px rgba(40, 167, 69, 0.3)'
                      : 'none',
                    background: selectedGenres.length > 0 && !loading
                      ? 'linear-gradient(135deg, #28a745, #20c997)'
                      : undefined,
                    transform: selectedGenres.length > 0 && !loading ? 'translateY(-2px)' : 'none'
                  }}
                >
                  {loading && (
                    <span
                      className="spinner-border spinner-border-sm me-2"
                      style={{ animation: 'spin 1s linear infinite' }}
                    />
                  )}
                  <span className="d-flex align-items-center justify-content-center">
                    {loading ? "Saving Your Preferences..." : "Start My Reading Journey"}
                    {!loading && selectedGenres.length > 0 && (
                      <span className="ms-2" style={{ fontSize: '1.2rem' }}>🚀</span>
                    )}
                  </span>
                  {selectedGenres.length > 0 && !loading && (
                    <div
                      className="position-absolute top-0 start-0 w-100 h-100"
                      style={{
                        background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%)',
                        borderRadius: 'inherit',
                        animation: 'shine 3s ease-in-out infinite',
                        pointerEvents: "none"
                      }}
                    />
                  )}
                </button>
                {selectedGenres.length === 0 && (
                  <p className="text-muted small mt-3 mb-0">
                    💡 Select at least one genre to continue
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
