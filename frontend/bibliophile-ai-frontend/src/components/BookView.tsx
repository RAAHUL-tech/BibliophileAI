import { useState, useEffect } from "react";
import { ReactReader } from "react-reader";
import { FaStar, FaRegStar } from "react-icons/fa";

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


  useEffect(() => {
    // Load bookmark status for the current user and book
    fetch(`http://localhost:8000/books/${book.id}/bookmark`, {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(res => res.json())
      .then(data => setBookmarked(data.bookmarked))
      .catch(() => setBookmarked(false));
  }, [book.id, token]);

  // Fetch reviews and ratings
  useEffect(() => {
    setLoadingReviews(true);
    fetch(`http://localhost:8000/books/${book.id}/reviews-ratings`, {
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
      const res = await fetch(`http://localhost:8000/books/${book.id}/review`, {
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
        fetch(`http://localhost:8000/books/${book.id}/reviews-ratings`, {
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
    const url = `http://localhost:8000/books/${book.id}/bookmark`;
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
    <div className="container my-4">
      <button className="btn btn-secondary mb-3" onClick={onBack}>
        ← Back
      </button>
      <button 
        className="btn btn-outline-warning"
        onClick={handleBookmarkToggle} 
        title={bookmarked ? "Remove bookmark" : "Add bookmark"}
        style={{ fontSize: 24 }}
      >
        {bookmarked ? <FaStar /> : <FaRegStar />}
      </button>
      {bookmarkMsg && <div className="small mt-2 text-info">{bookmarkMsg}</div>}

      <div className="card shadow-lg mx-auto" style={{ maxWidth: 900 }}>
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
            <div className="card-body">
              <h3>{book.title}</h3>
              <p>
                <strong>Authors:</strong> {book.authors.join(", ")}
              </p>
              <p>
                <strong>Genres:</strong> {book.categories.join(", ")}
              </p>
              <div>
                <strong>Rating:</strong> {avgRating ? avgRating.toFixed(2) : "N/A"} / 5
              </div>

              <div className="mt-3">
                <strong>Reviews:</strong>
                {loadingReviews ? (
                  <span className="ms-2">Loading...</span>
                ) : reviews.length === 0 ? (
                  <span className="ms-2 text-muted">No reviews yet.</span>
                ) : (
                  <ul className="list-unstyled">
                    {reviews.map((review, idx) => (
                      <li key={idx} className="border-bottom pb-2 mb-2">
                        <div>
                          <strong>{review.user}</strong>
                          <span className="ms-2 text-warning">({review.rating}⭐)</span>
                        </div>
                        <small>{review.text}</small>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              <form onSubmit={handleReviewSubmit} className="mt-3">
                <label className="form-label">Your Rating</label>
                <select
                  className="form-select mb-2"
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
                  className="form-control mb-2"
                  placeholder="Write your review..."
                  value={newReview}
                  onChange={(e) => setNewReview(e.target.value)}
                  required
                />
                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? "Submitting..." : "Submit Review"}
                </button>
                {reviewMsg && (
                  <div
                    className={`mt-2 small ${
                      reviewMsg.includes("submitted") ? "text-success" : "text-danger"
                    }`}
                  >
                    {reviewMsg}
                  </div>
                )}
              </form>
            </div>
          </div>
        </div>

       <div className="card-footer">
        <h5>Read the Book</h5>
        {book.id ? (
          <div style={{ height: "600px", border: "1px solid #ddd", background: "#fff" }}>
            <ReactReader
              url={`http://localhost:8000/proxy-epub/${book.id}/`}
              location={location}
              locationChanged={setLocation}
              showToc={false}
            />
          </div>
        ) : (
          <p>No EPUB available.</p>
        )}
      </div>
      </div>
    </div>
  );
}
