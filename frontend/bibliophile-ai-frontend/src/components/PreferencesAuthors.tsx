import { useState, useEffect } from "react";

type PreferencesAuthorsProps = {
  token: string;
  initialSelectedAuthors?: string[];
  onSave: (authors: string[]) => void;
  onBack: () => void;
  loading?: boolean;
};

export default function PreferencesAuthors({
  token,
  initialSelectedAuthors = [],
  onSave,
  onBack,
  loading = false,
}: PreferencesAuthorsProps) {
  const [allAuthors, setAllAuthors] = useState<string[]>([]);
  const [selectedAuthors, setSelectedAuthors] = useState<string[]>(initialSelectedAuthors);

  useEffect(() => {
    const controller = new AbortController();
    fetch("http://localhost:8080/api/v1/user/popular-authors", {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal
    })
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data.authors)) {
          setAllAuthors(data.authors);
        } else {
          setAllAuthors([]);
        }
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          console.error('Fetch failed:', err);
          setAllAuthors([]);
        }
      });
    return () => controller.abort();
  }, [token]);

  const toggleAuthor = (author: string) => {
    setSelectedAuthors((prevAuthors) =>
      prevAuthors.includes(author)
        ? prevAuthors.filter((a) => a !== author)
        : [...prevAuthors, author]
    );
  };

  return (
    <div
      className="min-vh-100 d-flex align-items-center justify-content-center position-relative"
      style={{
        background: 'linear-gradient(135deg, #764ba2 0%, #667eea 50%, #f093fb 100%)',
        padding: '2rem 0'
      }}
    >
      {/* Animated background pattern */}
      <div
        className="position-absolute w-100 h-100 opacity-10"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.4'%3E%3Ccircle cx='30' cy='30' r='4'/%3E%3Ccircle cx='15' cy='15' r='2'/%3E%3Ccircle cx='45' cy='45' r='2'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`,
          animation: 'float 20s ease-in-out infinite',
          pointerEvents: 'none'
        }}
      />

      {/* Floating decorative elements */}
      <div
        className="position-absolute"
        style={{ top: '10%', left: '5%', animation: 'bounce 4s infinite', pointerEvents: 'none' }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-3" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '2rem' }}>✍️</span>
        </div>
      </div>
      <div
        className="position-absolute"
        style={{ top: '20%', right: '8%', animation: 'bounce 4s infinite 1s', pointerEvents: 'none' }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-2" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '1.5rem' }}>📝</span>
        </div>
      </div>
      <div
        className="position-absolute"
        style={{ bottom: '15%', left: '8%', animation: 'bounce 4s infinite 2s', pointerEvents: 'none' }}
      >
        <div className="rounded-circle bg-white bg-opacity-15 p-2" style={{ backdropFilter: 'blur(10px)' }}>
          <span style={{ fontSize: '1.5rem' }}>📖</span>
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
                  background: 'linear-gradient(135deg, rgba(118, 75, 162, 0.1) 0%, rgba(102, 126, 234, 0.1) 100%)',
                  borderRadius: '2rem 2rem 0 0'
                }}
              >
                <div className="mb-3">
                  <span style={{ fontSize: '3rem', filter: 'drop-shadow(0 4px 8px rgba(0,0,0,0.1))' }}>
                    ✍️📚
                  </span>
                </div>
                <h2
                  className="display-6 fw-bold mb-2"
                  style={{
                    background: 'linear-gradient(135deg, #764ba2, #667eea)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                    backgroundClip: 'text'
                  }}
                >
                  Choose Your Favourite Authors
                </h2>
                <p className="lead text-muted mb-0">
                  Select authors whose work you love to tailor your recommendations
                </p>
                {selectedAuthors.length > 0 && (
                  <div className="mt-3">
                    <span
                      className="badge px-3 py-2 rounded-pill"
                      style={{
                        background: 'linear-gradient(135deg, #764ba2, #667eea)',
                        color: 'white',
                        fontSize: '0.9rem',
                        animation: 'pulse 2s infinite'
                      }}
                    >
                      {selectedAuthors.length} {selectedAuthors.length === 1 ? 'author' : 'authors'} selected
                    </span>
                  </div>
                )}
              </div>

              {/* Card Body */}
              <div className="card-body p-4">
                {allAuthors.length === 0 ? (
                  <div className="text-center py-4" style={{ color: '#764ba2' }}>
                    <div
                      className="spinner-border mb-3"
                      style={{ color: '#764ba2', animation: 'spin 1s linear infinite' }}
                    />
                    <p className="text-muted">Loading popular authors...</p>
                  </div>
                ) : (
                  <div className="row g-2">
                    {allAuthors.map((author, index) => (
                      <div key={author} className="col-6 col-md-4">
                        <button
                          type="button"
                          className={`btn w-100 border-2 position-relative overflow-hidden ${
                            selectedAuthors.includes(author) ? 'btn-primary shadow' : 'btn-outline-primary'
                          }`}
                          onClick={() => toggleAuthor(author)}
                          disabled={loading}
                          style={{
                            minHeight: '52px',
                            borderRadius: '0.75rem',
                            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                            transform: selectedAuthors.includes(author) ? 'translateY(-2px)' : 'none',
                            boxShadow: selectedAuthors.includes(author)
                              ? '0 6px 20px rgba(118, 75, 162, 0.35)'
                              : '0 2px 8px rgba(0, 0, 0, 0.08)',
                            animationDelay: `${index * 0.04}s`,
                            background: selectedAuthors.includes(author)
                              ? 'linear-gradient(135deg, #764ba2, #667eea)'
                              : 'transparent',
                            border: selectedAuthors.includes(author)
                              ? '2px solid transparent'
                              : '2px solid #764ba2',
                            fontSize: '0.82rem',
                            fontWeight: 600,
                            color: selectedAuthors.includes(author) ? 'white' : '#764ba2'
                          }}
                        >
                          {selectedAuthors.includes(author) && (
                            <div
                              className="position-absolute top-0 start-0 w-100 h-100"
                              style={{
                                background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.2) 50%, transparent 70%)',
                                borderRadius: 'inherit',
                                animation: 'shine 2.5s ease-in-out infinite',
                                pointerEvents: 'none'
                              }}
                            />
                          )}
                          {author}
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Card Footer */}
              <div
                className="card-footer border-0 text-center py-4"
                style={{
                  background: 'linear-gradient(135deg, rgba(118, 75, 162, 0.05) 0%, rgba(102, 126, 234, 0.05) 100%)',
                  borderRadius: '0 0 2rem 2rem'
                }}
              >
                <div className="d-flex justify-content-center gap-3">
                  <button
                    className="btn btn-outline-secondary btn-lg px-4 py-2 rounded-pill fw-bold"
                    onClick={onBack}
                    disabled={loading}
                    style={{ minWidth: '120px', transition: 'all 0.3s ease' }}
                  >
                    ← Back
                  </button>
                  <button
                    className={`btn btn-lg px-5 py-2 rounded-pill fw-bold position-relative overflow-hidden ${
                      selectedAuthors.length === 0 ? 'btn-outline-secondary' : 'btn-success'
                    }`}
                    disabled={selectedAuthors.length === 0 || loading}
                    onClick={() => onSave(selectedAuthors)}
                    style={{
                      minWidth: '160px',
                      transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
                      boxShadow: selectedAuthors.length > 0 && !loading
                        ? '0 8px 25px rgba(40, 167, 69, 0.3)'
                        : 'none',
                      background: selectedAuthors.length > 0 && !loading
                        ? 'linear-gradient(135deg, #28a745, #20c997)'
                        : undefined,
                      transform: selectedAuthors.length > 0 && !loading ? 'translateY(-2px)' : 'none'
                    }}
                  >
                    {loading && (
                      <span
                        className="spinner-border spinner-border-sm me-2"
                        style={{ animation: 'spin 1s linear infinite' }}
                      />
                    )}
                    <span>
                      {loading ? "Saving..." : "Next →"}
                    </span>
                    {selectedAuthors.length > 0 && !loading && (
                      <div
                        className="position-absolute top-0 start-0 w-100 h-100"
                        style={{
                          background: 'linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.3) 50%, transparent 70%)',
                          borderRadius: 'inherit',
                          animation: 'shine 3s ease-in-out infinite',
                          pointerEvents: 'none'
                        }}
                      />
                    )}
                  </button>
                </div>
                {selectedAuthors.length === 0 && (
                  <p className="text-muted small mt-3 mb-0">
                    💡 Select at least one author or skip to continue
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
