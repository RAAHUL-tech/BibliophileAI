import { type FC, useRef } from "react"
import { FaChevronLeft, FaChevronRight, FaTimes } from "react-icons/fa"
import BookOverlay, { type BookRecommendation } from "./BookOverlay"
import "./BookOverlay.css"
import "./NetflixStyles.css"

interface SearchOverlayProps {
  results: BookRecommendation[]
  loading: boolean
  error: string | null
  onClose: () => void
  onBookClick: (book: BookRecommendation) => void
}

const SearchOverlay: FC<SearchOverlayProps> = ({ results, loading, error, onClose, onBookClick }) => {
  const sliderRef = useRef<HTMLDivElement>(null)

  const scrollLeft = () => {
    if (sliderRef.current) {
      sliderRef.current.scrollBy({ left: -600, behavior: "smooth" })
    }
  }

  const scrollRight = () => {
    if (sliderRef.current) {
      sliderRef.current.scrollBy({ left: 600, behavior: "smooth" })
    }
  }

  return (
    <div className="bib-overlay-backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="bib-overlay-panel" onClick={(e) => e.stopPropagation()}>
        <button
          type="button"
          className="bib-overlay-close"
          onClick={onClose}
          aria-label="Close search results"
        >
          <FaTimes />
        </button>

        <div className="netflix-container">
          <h2 className="netflix-row-title">Search results</h2>
          {loading && (
            <p style={{ color: "var(--bib-text-muted)" }}>Searching...</p>
          )}
          {error && !loading && (
            <div className="bib-alert-danger my-3">{error}</div>
          )}
          {!loading && !error && results.length === 0 && (
            <p style={{ color: "var(--bib-text-muted)" }}>No books found for your search.</p>
          )}

          {!loading && !error && results.length > 0 && (
            <div className="netflix-row">
              <h3 className="netflix-row-title">Top matches</h3>
              <div className="netflix-row-content" style={{ position: "relative" }}>
                <button
                  className="netflix-scroll-button left"
                  onClick={scrollLeft}
                  aria-label="Scroll left"
                >
                  <FaChevronLeft />
                </button>
                <div className="netflix-slider" ref={sliderRef}>
                  {results.map((book) => (
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
          )}
        </div>
      </div>
    </div>
  )
}

export default SearchOverlay

