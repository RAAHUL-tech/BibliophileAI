import { useState, useEffect, useRef } from "react";
import { ReactReader } from "react-reader";
import { FaStar, FaRegStar } from "react-icons/fa";
import "./SharedStyles.css";

interface BookRecommendation {
  id: string;
  title: string;
  authors: string[];
  categories: string[];
  thumbnail_url?: string;
  download_link?: string;
}

interface Review {
  user: string;
  rating: number;
  text: string;
}

interface BookViewProps {
  book: BookRecommendation;
  token: string;
  onBack: () => void;
}

export default function BookView({ book, token, onBack }: BookViewProps) {
  const [reviews, setReviews] = useState<Review[]>([]);
  const [avgRating, setAvgRating] = useState<number>(0);
  const [loadingReviews, setLoadingReviews] = useState(true);
  const [newRating, setNewRating] = useState<number>(0);
  const [newReview, setNewReview] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [reviewMsg, setReviewMsg] = useState<string | null>(null);
  const [location, setLocation] = useState<string | number>(0);
  const [bookmarked, setBookmarked] = useState(false);
  const [bookmarkMsg, setBookmarkMsg] = useState<string | null>(null);
  const hasReportedRead = useRef(false);
  const lastSentPage = useRef<string | number>(null);


  useEffect(() => {
    // Load bookmark status for the current user and book
    fetch(`http://localhost:8080/api/v1/user/books/${book.id}/bookmark`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => res.json())
      .then(data => setBookmarked(data.bookmarked))
      .catch(() => setBookmarked(false));
  }, [book.id, token]);

  // Fetch reviews and ratings
  useEffect(() => {
    setLoadingReviews(true);
    fetch(`http://localhost:8080/api/v1/user/books/${book.id}/reviews-ratings`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((res) => res.json())
      .then((data) => {
        setReviews(data.reviews || []);
        setAvgRating(data.avg_rating || 0);
      })
      .catch(() => setReviews([]))
      .finally(() => setLoadingReviews(false));
  }, [book.id, token]);

 useEffect(() => {
  if (!hasReportedRead.current && book.id && token) {
    fetch(`http://localhost:8080/api/v1/user/books/${book.id}/track-read`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`
      }
    });
    hasReportedRead.current = true;
  }
}, [book.id, token]);

useEffect(() => {
    if (!token || !book.id) return;
    if (location === lastSentPage.current) return;

    // Send page turn event on location change
    fetch(`http://localhost:8080/api/v1/user/books/${book.id}/page-turn`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ page: location }), // Send current page
    }).catch((e) => {
      console.error("Failed to record page turn:", e);
    });

    lastSentPage.current = location;
  }, [location, token, book.id]);
  
  // Submit new review
  const handleReviewSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newRating || !newReview.trim()) {
      setReviewMsg("Please enter both rating and review.");
      return;
    }
    setSubmitting(true);
    setReviewMsg(null);
    try {
      const res = await fetch(`http://localhost:8080/api/v1/user/books/${book.id}/review`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({ rating: newRating, text: newReview }),
      });
      if (res.ok) {
        setReviewMsg("Review submitted!");
        setNewRating(0);
        setNewReview("");
        // Refetch all reviews to avoid stale state and ensure DB consistency
        fetch(`http://localhost:8080/api/v1/user/books/${book.id}/reviews-ratings`, {
        headers: { Authorization: `Bearer ${token}` }
        })
        .then((res) => res.json())
        .then((data) => {
            setReviews(data.reviews || []);
            setAvgRating(data.avg_rating || 0);
            setReviewMsg("Review submitted!");
        })
        .catch(() => setReviewMsg("Review submitted, but failed to update list."));

      } else {
        const err = await res.json();
        setReviewMsg(err.detail || "Failed to submit review.");
      }
    } catch {
      setReviewMsg("Network error while submitting review.");
    }
    setSubmitting(false);
  };


  const handleBookmarkToggle = async () => {
    const url = `http://localhost:8080/api/v1/user/books/${book.id}/bookmark`;
    try {
      const res = await fetch(url, {
        method: bookmarked ? "DELETE" : "POST",
        headers: { Authorization: `Bearer ${token}` }
      });
      if (res.ok) {
        setBookmarked(!bookmarked);
        setBookmarkMsg(bookmarked ? "Bookmark removed!" : "Bookmarked!");
      } else {
        setBookmarkMsg("Failed to update bookmark.");
      }
    } catch {
      setBookmarkMsg("Network error.");
    }
  };


  return (
    <div className="container my-4" style={{ color: "var(--bib-text)" }}>
      <div className="d-flex align-items-center gap-2 mb-3 flex-wrap">
        <button type="button" className="bib-btn-secondary" onClick={onBack}>
          ← Back
        </button>
        <button
          type="button"
          className="bib-btn-secondary"
          onClick={handleBookmarkToggle}
          title={bookmarked ? "Remove bookmark" : "Add bookmark"}
          style={{ fontSize: "1.25rem", padding: "0.5rem 0.75rem" }}
        >
          {bookmarked ? <FaStar style={{ color: "var(--bib-accent)" }} /> : <FaRegStar />}
        </button>
        {bookmarkMsg && (
          <span className="small" style={{ color: "var(--bib-text-muted)" }}>{bookmarkMsg}</span>
        )}
      </div>

      <div
        className="shadow-lg mx-auto rounded-3 overflow-hidden"
        style={{
          maxWidth: 900,
          background: "var(--bib-bg-elevated)",
          border: "1px solid var(--bib-border)",
        }}
      >
        <div className="row g-0">
          <div className="col-md-4">
            {book.thumbnail_url && (
              <img
                src={book.thumbnail_url}
                alt={book.title}
                className="img-fluid rounded-start"
                style={{ maxHeight: 300, objectFit: "cover" }}
              />
            )}
          </div>
          <div className="col-md-8">
            <div className="p-4" style={{ color: "var(--bib-text)" }}>
              <h3 style={{ color: "var(--bib-text-title)" }}>{book.title}</h3>
              <p><strong>Authors:</strong> {book.authors.join(", ")}</p>
              <p><strong>Genres:</strong> {book.categories.join(", ")}</p>
              <p><strong>Rating:</strong> {avgRating ? avgRating.toFixed(2) : "N/A"} / 5</p>

              <div className="mt-3">
                <strong>Reviews:</strong>
                {loadingReviews ? (
                  <span className="ms-2" style={{ color: "var(--bib-text-muted)" }}>Loading...</span>
                ) : reviews.length === 0 ? (
                  <span className="ms-2" style={{ color: "var(--bib-text-muted)" }}>No reviews yet.</span>
                ) : (
                  <ul className="list-unstyled mt-2">
                    {reviews.map((review, idx) => (
                      <li key={idx} className="pb-2 mb-2 border-bottom" style={{ borderBottomColor: "var(--bib-border)" }}>
                        <div>
                          <strong>{review.user}</strong>
                          <span className="ms-2" style={{ color: "var(--bib-accent)" }}>({review.rating}⭐)</span>
                        </div>
                        <small style={{ color: "var(--bib-text-muted)" }}>{review.text}</small>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <form onSubmit={handleReviewSubmit} className="mt-3">
                <label className="form-label" style={{ color: "var(--bib-text)" }}>Your Rating</label>
                <select
                  className="bib-input form-select mb-2 w-100"
                  value={newRating}
                  onChange={(e) => setNewRating(Number(e.target.value))}
                  required
                >
                  <option value={0}>Select rating...</option>
                  <option value={1}>1 - Poor</option>
                  <option value={2}>2 - Fair</option>
                  <option value={3}>3 - Good</option>
                  <option value={4}>4 - Very Good</option>
                  <option value={5}>5 - Excellent</option>
                </select>
                <input
                  type="text"
                  className="bib-input form-control mb-2"
                  placeholder="Write your review..."
                  value={newReview}
                  onChange={(e) => setNewReview(e.target.value)}
                  required
                />
                <button type="submit" className="bib-btn-primary" disabled={submitting}>
                  {submitting ? "Submitting..." : "Submit Review"}
                </button>
                {reviewMsg && (
                  <div
                    className="mt-2 small"
                    style={{ color: reviewMsg.includes("submitted") ? "var(--bib-accent)" : "var(--bib-text-muted)" }}
                  >
                    {reviewMsg}
                  </div>
                )}
              </form>
            </div>
          </div>
        </div>

        <div className="p-3 border-top" style={{ borderColor: "var(--bib-border)", background: "var(--bib-card-header)" }}>
          <h5 style={{ color: "var(--bib-text-title)" }}>Read the Book</h5>
          {book.id ? (
            <div style={{ height: "600px", border: "1px solid var(--bib-border)", background: "var(--bib-bg-elevated)", borderRadius: 8 }}>
              <ReactReader
                url={`http://localhost:8080/api/v1/user/proxy-epub/${book.id}/`}
                location={location}
                locationChanged={setLocation}
                showToc={false}
              />
            </div>
          ) : (
            <p style={{ color: "var(--bib-text-muted)" }}>No EPUB available.</p>
          )}
        </div>
      </div>
    </div>
  );
}
