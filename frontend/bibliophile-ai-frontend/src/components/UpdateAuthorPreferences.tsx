import { useState, useEffect } from "react"
import "./SharedStyles.css"

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

  useEffect(() => {
    setSelected(initialAuthors)
  }, [initialAuthors])

  useEffect(() => {
    const controller = new AbortController()
    fetch("http://localhost:8080/api/v1/user/popular-authors", {
      headers: { Authorization: `Bearer ${token}` },
      signal: controller.signal
    })
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data.authors)) setAllAuthors(data.authors)
        else setAllAuthors([])
      })
      .catch((err) => {
        if (err.name !== "AbortError") {
          console.error("Fetch failed:", err)
          setAllAuthors([])
        }
      })
    return () => controller.abort()
  }, [token])

  const toggleAuthor = (author: string) => {
    setSelected(prev =>
      prev.includes(author)
        ? prev.filter(a => a !== author)
        : [...prev, author]
    )
  }

  return (
    <div className="container my-5">
      <div className="bib-modal" style={{ maxWidth: 560 }}>
        <div className="bib-modal-header">Update Author Preferences</div>
        <div className="bib-modal-body">
          <p className="text-muted mb-3" style={{ color: "var(--bib-text-muted)" }}>
            Select your favorite authors to personalize recommendations
          </p>
          <div className="bib-chip-wrap mb-3">
            {allAuthors.map((author) => (
              <button
                key={author}
                type="button"
                className={`bib-chip ${selected.includes(author) ? "bib-chip-selected" : ""}`}
                onClick={() => toggleAuthor(author)}
                disabled={loading}
              >
                {author}
              </button>
            ))}
          </div>
          {allAuthors.length === 0 && !loading && (
            <div className="bib-alert-info mb-3">No popular authors available at the moment.</div>
          )}
          <div className="d-flex gap-2">
            <button
              type="button"
              className="bib-btn-primary"
              disabled={loading || selected.length === 0}
              onClick={() => onSave(selected)}
            >
              {loading ? "Saving..." : "Update"}
            </button>
            <button type="button" className="bib-btn-secondary" onClick={onCancel} disabled={loading}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
