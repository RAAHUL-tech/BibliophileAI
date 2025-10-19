import { useState, useEffect } from "react"

interface UpdateAuthorPreferencesProps {
  initialAuthors: string[]
  loading: boolean
  onSave: (authors: string[]) => void
  onCancel: () => void
  token: string
}

export default function UpdateAuthorPreferences({
  initialAuthors, 
  loading, 
  onSave, 
  onCancel,
  token
}: UpdateAuthorPreferencesProps) {
  const [allAuthors, setAllAuthors] = useState<string[]>([])
  const [selected, setSelected] = useState<string[]>(initialAuthors)

  // Sync selected authors when initialAuthors changes
  useEffect(() => {
    setSelected(initialAuthors)
  }, [initialAuthors])

  // Fetch popular authors from backend
  useEffect(() => {
    const controller = new AbortController();
    fetch("http://localhost:8000/api/v1/user/popular-authors", {
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
    setSelected(prev =>
      prev.includes(author)
        ? prev.filter(a => a !== author)
        : [...prev, author]
    )
  }

  return (
    <div className="container my-5">
      <div className="card shadow-lg mx-auto" style={{ maxWidth: 600 }}>
        <div className="card-header text-center">
          <h4>Update Author Preferences</h4>
        </div>
        <div className="card-body">
          <p className="text-muted mb-3">
            Select your favorite authors to personalize recommendations
          </p>
          
          <div className="row g-2 mb-3">
            {allAuthors.map(author => (
              <div className="col-6 col-md-4" key={author}>
                <button
                  className={`btn w-100 mb-2 ${
                    selected.includes(author) 
                      ? "btn-primary" 
                      : "btn-outline-primary"
                  }`}
                  onClick={() => toggleAuthor(author)}
                  type="button"
                  disabled={loading}
                >
                  {author}
                </button>
              </div>
            ))}
          </div>
          
          {allAuthors.length === 0 && !loading && (
            <div className="alert alert-info">
              No popular authors available at the moment.
            </div>
          )}
          
          <div className="d-flex gap-2">
            <button 
              className="btn btn-success"
              disabled={loading || selected.length === 0}
              onClick={() => onSave(selected)}
            >
              {loading ? "Saving..." : "Update"}
            </button>
            <button 
              className="btn btn-secondary"
              onClick={onCancel}
              disabled={loading}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
