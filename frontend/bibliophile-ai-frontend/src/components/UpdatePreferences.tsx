import { useState, useEffect } from "react"

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

export default function UpdatePreferences({
  initialGenres, loading, onSave, onCancel
}: UpdatePreferencesProps) {
  const [selected, setSelected] = useState<string[]>(initialGenres)

  // Add this effect to update selected whenever initialGenres changes
  useEffect(() => {
    setSelected(initialGenres)
  }, [initialGenres])

  const toggleGenre = (genre: string) => {
    setSelected(selected =>
      selected.includes(genre)
        ? selected.filter(g => g !== genre)
        : [...selected, genre]
    )
  }

  return (
    <div className="container my-5">
      <div className="card shadow-lg mx-auto" style={{ maxWidth: 600 }}>
        <div className="card-header text-center"><h4>Update Preferences</h4></div>
        <div className="card-body">
          <div className="row g-2 mb-3">
            {allGenres.map(genre => (
              <div className="col-6 col-md-4" key={genre}>
                <button
                  className={`btn w-100 mb-2 ${selected.includes(genre) ? "btn-primary" : "btn-outline-primary"}`}
                  onClick={() => toggleGenre(genre)}
                  type="button"
                >
                  {genre}
                </button>
              </div>
            ))}
          </div>
          <button className="btn btn-success me-2" disabled={loading || selected.length === 0} onClick={() => onSave(selected)}>
            {loading ? "Saving..." : "Update"}
          </button>
          <button className="btn btn-secondary" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  )
}
