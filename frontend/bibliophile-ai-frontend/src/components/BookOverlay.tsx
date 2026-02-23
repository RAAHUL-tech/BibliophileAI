import { useState, useCallback, type FC } from "react"
import BookView from "./BookView"
import { FaTimes, FaBookOpen } from "react-icons/fa"
import "./BookOverlay.css"

export interface BookRecommendation {
  id: string
  title: string
  authors: string[]
  categories: string[]
  thumbnail_url?: string
  download_link?: string
  score?: number
}

interface BookOverlayProps {
  book: BookRecommendation
  token: string
  onClose: () => void
}

const BookOverlay: FC<BookOverlayProps> = ({ book, token, onClose }) => {
  const [mode, setMode] = useState<"preview" | "reading">("preview")

  const handleRead = useCallback(() => setMode("reading"), [])
  const handleBack = useCallback(() => {
    setMode("preview")
  }, [])
  const handleClose = useCallback(() => {
    if (mode === "reading") setMode("preview")
    else onClose()
  }, [mode, onClose])

  return (
    <div className="bib-overlay-backdrop" onClick={handleClose} role="dialog" aria-modal="true">
      <div
        className="bib-overlay-panel"
        onClick={(e) => e.stopPropagation()}
      >
        <button
          type="button"
          className="bib-overlay-close"
          onClick={handleClose}
          aria-label="Close"
        >
          <FaTimes />
        </button>

        {mode === "preview" ? (
          <div className="bib-overlay-preview">
            <div className="bib-overlay-preview-hero">
              {book.thumbnail_url ? (
                <img src={book.thumbnail_url} alt={book.title} className="bib-overlay-poster" />
              ) : (
                <div className="bib-overlay-poster-placeholder">
                  <span>{book.title}</span>
                </div>
              )}
              <div className="bib-overlay-preview-info">
                <h1 className="bib-overlay-title">{book.title}</h1>
                {book.authors?.length > 0 && (
                  <p className="bib-overlay-meta">By {book.authors.join(", ")}</p>
                )}
                {book.categories?.length > 0 && (
                  <p className="bib-overlay-meta">{book.categories.join(" · ")}</p>
                )}
                <div className="bib-overlay-actions">
                  <button type="button" className="bib-btn bib-btn-primary" onClick={handleRead}>
                    <FaBookOpen className="me-2" /> Read book
                  </button>
                  <button type="button" className="bib-btn bib-btn-secondary" onClick={onClose}>
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="bib-overlay-reading">
            <BookView book={book} token={token} onBack={handleBack} />
          </div>
        )}
      </div>
    </div>
  )
}

export default BookOverlay
