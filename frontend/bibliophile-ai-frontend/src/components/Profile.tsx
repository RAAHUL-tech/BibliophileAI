import { useEffect, useState } from "react";
import BookView from "./BookView"; 

interface ProfileProps {
  token: string;
  onClose: () => void;
}

interface UserProfile {
  username: string;
  email: string;
  age: number | null;
  pincode: string | null;
}

interface BookRecommendation {
  id: string;
  title: string;
  authors: string[];
  categories: string[];
  thumbnail_url?: string;
  download_link?: string;
}

export default function Profile({ token, onClose }: ProfileProps) {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [bookmarks, setBookmarks] = useState<BookRecommendation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [age, setAge] = useState<string>("");
  const [pincode, setPincode] = useState<string>("");
  const [selectedBook, setSelectedBook] = useState<BookRecommendation | null>(null); // Local modal state

  // Fetch user profile
  useEffect(() => {
    const fetchProfile = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("http://localhost:8000/user/profile", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          setProfile(data);
          setAge(data.age?.toString() || "");
          setPincode(data.pincode || "");
        } else {
          setError("Failed to load profile");
        }
      } catch {
        setError("Network error while loading profile");
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, [token]);

  // Fetch bookmarked books
  useEffect(() => {
    const fetchBookmarks = async () => {
      try {
        const res = await fetch("http://localhost:8000/user/bookmarks", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          setBookmarks(data.bookmarks || []);
        } else {
          console.error("Failed to load bookmarks");
        }
      } catch (err) {
        console.error("Network error loading bookmarks:", err);
      }
    };
    if (!isEditing) {
      fetchBookmarks();
    }
  }, [token, isEditing]); // Re-fetch when editing ends

  const handleEdit = () => {
    setIsEditing(true);
  };

  const handleSave = async () => {
    if (!age || !pincode) {
      setError("Age and pincode are required.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await fetch("http://localhost:8000/user/profile_update", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          genres: [],
          authors: [],
          age: Number(age),
          pincode,
        }),
      });

      if (res.ok) {
        const updated = await res.json();
        setProfile((prev) =>
          prev ? { ...prev, age: updated.age, pincode: updated.pincode } : null
        );
        setIsEditing(false);
      } else {
        const err = await res.json();
        setError(err.detail || "Failed to save profile");
      }
    } catch {
      setError("Network error while saving profile");
    } finally {
      setLoading(false);
    }
  };

  const handleCancel = () => {
    setAge(profile?.age?.toString() || "");
    setPincode(profile?.pincode || "");
    setIsEditing(false);
    setError(null);
  };

  // Close BookView and return to profile
  const closeBookView = () => {
    setSelectedBook(null);
  };

  return (
    <div className="container my-4">
      {selectedBook ? (
        // Render BookView when a book is selected
        <BookView book={selectedBook} token={token} onBack={closeBookView} />
      ) : (
        <div className="card shadow-lg mx-auto" style={{ maxWidth: 800 }}>
          <div className="card-header text-center bg-primary text-white">
            <h4>User Profile</h4>
          </div>
          <div className="card-body">
            {loading ? (
              <p className="text-center">Loading...</p>
            ) : error ? (
              <p className="text-danger">{error}</p>
            ) : profile ? (
              <>
                <p>
                  <b>Name:</b> {profile.username}
                </p>
                <p>
                  <b>Email:</b> {profile.email}
                </p>

                {isEditing ? (
                  <>
                    <div className="mb-3">
                      <label className="form-label">Age</label>
                      <input
                        type="number"
                        className="form-control"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        min="10"
                        max="120"
                        required
                      />
                    </div>
                    <div className="mb-3">
                      <label className="form-label">Pincode</label>
                      <input
                        type="text"
                        className="form-control"
                        value={pincode}
                        onChange={(e) => setPincode(e.target.value)}
                        minLength={5}
                        maxLength={10}
                        required
                      />
                    </div>
                    <div className="d-flex gap-2 mb-4">
                      <button
                        className="btn btn-success"
                        onClick={handleSave}
                        disabled={loading}
                      >
                        {loading ? "Saving..." : "Save"}
                      </button>
                      <button
                        className="btn btn-secondary"
                        onClick={handleCancel}
                        disabled={loading}
                      >
                        Cancel
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <p>
                      <b>Age:</b> {profile.age || "Not specified"}
                    </p>
                    <p>
                      <b>Pincode:</b> {profile.pincode || "Not specified"}
                    </p>
                    <button
                      className="btn btn-outline-primary w-100 mb-4"
                      onClick={handleEdit}
                    >
                      Edit Profile
                    </button>
                  </>
                )}

                {/* Bookmarked Books Grid */}
                {bookmarks.length > 0 && (
                  <div className="mt-4">
                    <h5>⭐ Your Bookmarks</h5>
                    <div className="row">
                      {bookmarks.map((book) => (
                        <div key={book.id} className="col-md-4 col-lg-3 mb-4">
                          <div
                            className="card h-100"
                            style={{ cursor: "pointer" }}
                            onClick={() => setSelectedBook(book)}
                          >
                            {book.thumbnail_url && (
                              <img
                                src={book.thumbnail_url}
                                className="card-img-top"
                                alt={book.title}
                                style={{ height: "200px", objectFit: "cover" }}
                              />
                            )}
                            <div className="card-body d-flex flex-column">
                              <h6 className="card-title mb-1">{book.title}</h6>
                              <p className="card-text text-muted mb-1">
                                by {book.authors.join(", ")}
                              </p>
                              <p className="card-text small text-secondary mt-auto">
                                {book.categories.slice(0, 2).join(", ")}
                              </p>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <button
                  className="btn btn-secondary w-100"
                  onClick={onClose}
                  disabled={loading}
                >
                  Close
                </button>
              </>
            ) : (
              <p className="text-warning">No profile data.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
