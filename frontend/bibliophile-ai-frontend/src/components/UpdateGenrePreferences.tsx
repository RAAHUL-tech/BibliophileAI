import { useState, useEffect } from "react"
import "./SharedStyles.css"

const allGenres = [
  "Fantasy", "Romance", "Science Fiction", "Mystery", "Thriller",
  "Non-fiction", "Historical", "Young Adult", "Horror", "Biography"
]

interface UpdatePreferencesProps {
  initialGenres: string[]
  loading: boolean
  onSave: (genres: string[]) => void
  onCancel: () => void
}

export default function UpdateGenrePreferences({
  initialGenres, loading, onSave, onCancel
}: UpdatePreferencesProps) {
  const [selected, setSelected] = useState<string[]>(initialGenres)

  useEffect(() => {
    setSelected(initialGenres)
  }, [initialGenres])

  const toggleGenre = (genre: string) => {
    setSelected(prev =>
      prev.includes(genre)
        ? prev.filter(g => g !== genre)
        : [...prev, genre]
    )
  }

  return (
    <div className="container my-5">
      <div className="bib-modal" style={{ maxWidth: 560 }}>
        <div className="bib-modal-header">Update Preferences</div>
        <div className="bib-modal-body">
          <p className="text-muted mb-3" style={{ color: "var(--bib-text-muted)" }}>
            Choose genres you enjoy for better recommendations
          </p>
          <div className="bib-chip-wrap mb-4">
            {allGenres.map((genre) => (
              <button
                key={genre}
                type="button"
                className={`bib-chip ${selected.includes(genre) ? "bib-chip-selected" : ""}`}
                onClick={() => toggleGenre(genre)}
              >
                {genre}
              </button>
            ))}
          </div>
          <div className="d-flex gap-2">
            <button
              type="button"
              className="bib-btn-primary"
              disabled={loading || selected.length === 0}
              onClick={() => onSave(selected)}
            >
              {loading ? "Saving..." : "Update"}
            </button>
            <button type="button" className="bib-btn-secondary" onClick={onCancel}>
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}
