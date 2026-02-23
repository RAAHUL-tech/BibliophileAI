import { useEffect, useState, useRef } from "react";
import BookView from "./BookView";
import { FaChevronLeft, FaChevronRight } from "react-icons/fa";
import "./SharedStyles.css";
import "./NetflixStyles.css";

interface ProfileProps {
  token: string;
  onClose: () => void;
}

interface UserProfile {
  id: string;
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

interface User {
  id: string;
  username: string;
}

export default function Profile({ token, onClose }: ProfileProps) {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [bookmarks, setBookmarks] = useState<BookRecommendation[]>([]);
  const [followers, setFollowers] = useState<User[]>([]);
  const [following, setFollowing] = useState<User[]>([]);
  const [activeTab, setActiveTab] = useState<"following" | "followers">("following");
  const [viewingFollowersOf, setViewingFollowersOf] = useState<string | null>(null);
  const [viewingFollowersList, setViewingFollowersList] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isEditing, setIsEditing] = useState(false);
  const [age, setAge] = useState<string>("");
  const [pincode, setPincode] = useState<string>("");
  const [selectedBook, setSelectedBook] = useState<BookRecommendation | null>(null);
  const [followingUserIds, setFollowingUserIds] = useState<Set<string>>(new Set());


  // Fetch user profile
  useEffect(() => {
    const fetchProfile = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await fetch("http://localhost:8080/api/v1/user/profile", {
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

  // Fetch followers and following
  useEffect(() => {
    const fetchFollowersAndFollowing = async () => {
      try {
        const res = await fetch("http://localhost:8080/api/v1/user/my-followers", {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (res.ok) {
          const data = await res.json();
          setFollowers(data.followers || []);
          setFollowing(data.following || []);
          // Store ids of following users for fast lookup
        const followingIds = new Set<string>((data.following || []).map((f: User) => f.id));
        setFollowingUserIds(followingIds);

        } else {
          console.error("Failed to load followers/following");
        }
      } catch (err) {
        console.error("Network error loading followers/following:", err);
      }
    };
    if (!isEditing && !viewingFollowersOf) {
      fetchFollowersAndFollowing();
    }
  }, [token, isEditing, viewingFollowersOf]);
  // Fetch bookmarked books
  useEffect(() => {
    const fetchBookmarks = async () => {
      try {
        const res = await fetch("http://localhost:8080/api/v1/user/bookmarks", {
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
    if (!isEditing && !viewingFollowersOf) {
      fetchBookmarks();
    }
  }, [token, isEditing, viewingFollowersOf]);

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
      const res = await fetch("http://localhost:8080/user/profile_update", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          age: Number(age),
          pincode,
        }),
      });

      if (res.ok) {
        const updated = await res.json();
        setProfile((prev) => (prev ? { ...prev, age: updated.age, pincode: updated.pincode } : null));
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

  const handleUnfollow = async (targetUserId: string) => {
    try {
      const res = await fetch(`http://localhost:8080/api/v1/user/follow/${targetUserId}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        if (activeTab === "followers") {
          setFollowers((prev) => prev.filter((f) => f.id !== targetUserId));
        } else {
          setFollowing((prev) => prev.filter((f) => f.id !== targetUserId));
        }
        alert("Unfollowed successfully");
      } else {
        alert("Failed to unfollow");
      }
    } catch (err) {
      console.error("Error unfollowing:", err);
      alert("Network error while unfollowing");
    }
  };

  // Follow back button handler
  const handleFollow = async (targetUserId: string) => {
    try {
      const res = await fetch(`http://localhost:8080/api/v1/user/follow/${targetUserId}`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        alert("Followed successfully");
        // Optionally refresh followers/following lists here
      } else {
        alert("Failed to follow");
      }
    } catch (err) {
      console.error("Error following user:", err);
      alert("Network error while following user");
    }
  };

  const handleViewFollowersList = async (userId: string) => {
    try {
      const res = await fetch(`http://localhost:8080/api/v1/user/followers/${userId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.ok) {
        const data = await res.json();
        if(data.count==0)  alert("Can't display because either you or the other is not following you");
        setViewingFollowersList(data.followers || []);
        setViewingFollowersOf(userId);
      } else if (res.status === 403) {
        alert("Can't display because either you or the other is not following you");
      } else {
        alert("Failed to load followers");
      }
    } catch (err) {
      console.error("Error fetching followers:", err);
      alert("Network error while loading followers");
    }
  };

  const closeFollowersView = () => {
    setViewingFollowersOf(null);
    setViewingFollowersList([]);
  };

  const closeBookView = () => {
    setSelectedBook(null);
  };

  // Display a viewed followers list of another user
  if (viewingFollowersOf) {
    return (
      <div className="container my-4">
        <div className="bib-modal">
          <div className="bib-modal-header">Followers List</div>
          <div className="bib-modal-body">
            <button type="button" className="bib-btn-secondary mb-3" onClick={closeFollowersView}>
              ← Back to Profile
            </button>
            {viewingFollowersList.length > 0 ? (
              <div>
                {viewingFollowersList.map((follower) => (
                  <div key={follower.id} className="bib-list-item">
                    <span>{follower.username}</span>
                    {!followingUserIds.has(follower.id) && follower.id !== profile?.id && (
                      <button type="button" className="bib-btn-primary" style={{ padding: "0.35rem 0.75rem", fontSize: "0.875rem" }} onClick={() => handleFollow(follower.id)}>
                        Follow Back
                      </button>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted">No followers found.</p>
            )}
          </div>
        </div>
      </div>
    );
  }

  const bookmarksSliderRef = useRef<HTMLDivElement>(null);

  const scrollBookmarks = (dir: "left" | "right") => {
    if (!bookmarksSliderRef.current) return;
    bookmarksSliderRef.current.scrollBy({ left: dir === "left" ? -320 : 320, behavior: "smooth" });
  };

  // Main Profile View
  return (
    <div className="container my-4">
      {selectedBook ? (
        <BookView book={selectedBook} token={token} onBack={closeBookView} />
      ) : (
        <div className="bib-modal">
          <div className="bib-modal-header">User Profile</div>
          <div className="bib-modal-body">
            {loading ? (
              <p className="text-center" style={{ color: "var(--bib-text-muted)" }}>Loading...</p>
            ) : error ? (
              <p className="bib-alert-danger">{error}</p>
            ) : profile ? (
              <>
                <p><b>Name:</b> {profile.username}</p>
                <p><b>Email:</b> {profile.email}</p>

                {isEditing ? (
                  <>
                    <div className="mb-3">
                      <label className="form-label" style={{ color: "var(--bib-text)" }}>Age</label>
                      <input
                        type="number"
                        className="bib-input form-control"
                        value={age}
                        onChange={(e) => setAge(e.target.value)}
                        min={10}
                        max={120}
                        required
                      />
                    </div>
                    <div className="mb-3">
                      <label className="form-label" style={{ color: "var(--bib-text)" }}>Pincode</label>
                      <input
                        type="text"
                        className="bib-input form-control"
                        value={pincode}
                        onChange={(e) => setPincode(e.target.value)}
                        minLength={5}
                        maxLength={10}
                        required
                      />
                    </div>
                    <div className="d-flex gap-2 mb-4">
                      <button type="button" className="bib-btn-primary" onClick={handleSave} disabled={loading}>
                        {loading ? "Saving..." : "Save"}
                      </button>
                      <button type="button" className="bib-btn-secondary" onClick={handleCancel} disabled={loading}>
                        Cancel
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <p><b>Age:</b> {profile.age ?? "Not specified"}</p>
                    <p><b>Pincode:</b> {profile.pincode ?? "Not specified"}</p>

                    <div className="bib-tabs">
                      <button
                        type="button"
                        className={`bib-tab ${activeTab === "following" ? "bib-tab-active" : ""}`}
                        onClick={() => setActiveTab("following")}
                      >
                        Following ({following.length})
                      </button>
                      <button
                        type="button"
                        className={`bib-tab ${activeTab === "followers" ? "bib-tab-active" : ""}`}
                        onClick={() => setActiveTab("followers")}
                      >
                        Followers ({followers.length})
                      </button>
                    </div>
                    <div>
                      {activeTab === "following" ? (
                        <div className="mb-4">
                          {following.map((user) => (
                            <div key={user.id} className="bib-list-item">
                              <span className="bib-link" onClick={() => handleViewFollowersList(user.id)}>
                                {user.username}
                              </span>
                              <button type="button" className="bib-btn-danger-outline" onClick={() => handleUnfollow(user.id)}>
                                Unfollow
                              </button>
                            </div>
                          ))}
                          {following.length === 0 && <p className="text-muted">No following users.</p>}
                        </div>
                      ) : (
                        <div className="mb-4">
                          {followers.map((user) => (
                            <div key={user.id} className="bib-list-item">
                              <span className="bib-link" onClick={() => handleViewFollowersList(user.id)}>
                                {user.username}
                              </span>
                              {!followingUserIds.has(user.id) && user.id !== profile?.id && (
                                <button type="button" className="bib-btn-primary" style={{ padding: "0.35rem 0.75rem", fontSize: "0.875rem" }} onClick={() => handleFollow(user.id)}>
                                  Follow Back
                                </button>
                              )}
                            </div>
                          ))}
                          {followers.length === 0 && <p className="text-muted">No followers found.</p>}
                        </div>
                      )}
                    </div>

                    <button type="button" className="bib-btn-secondary w-100 mb-4" onClick={handleEdit}>
                      Edit Profile
                    </button>
                  </>
                )}

                {bookmarks.length > 0 && (
                  <div className="mt-4">
                    <h5 style={{ color: "var(--bib-text-title)", marginBottom: "0.75rem" }}>⭐ Your Bookmarks</h5>
                    <div className="netflix-row-content" style={{ position: "relative" }}>
                      <button type="button" className="netflix-scroll-button left" onClick={() => scrollBookmarks("left")} aria-label="Scroll left">
                        <FaChevronLeft />
                      </button>
                      <div className="netflix-slider" ref={bookmarksSliderRef} style={{ padding: "1rem 56px" }}>
                        {bookmarks.map((book) => (
                          <div
                            key={book.id}
                            className="netflix-item"
                            onClick={() => setSelectedBook(book)}
                            onKeyDown={(e) => e.key === "Enter" && setSelectedBook(book)}
                            role="button"
                            tabIndex={0}
                          >
                            {book.thumbnail_url ? (
                              <img src={book.thumbnail_url} className="netflix-item-image" alt={book.title} />
                            ) : (
                              <div className="netflix-item-placeholder"><span>{book.title}</span></div>
                            )}
                            <div className="netflix-item-overlay">
                              <h4>{book.title}</h4>
                              <p>{book.authors?.join(", ")}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                      <button type="button" className="netflix-scroll-button right" onClick={() => scrollBookmarks("right")} aria-label="Scroll right">
                        <FaChevronRight />
                      </button>
                    </div>
                  </div>
                )}

                <button type="button" className="bib-btn-secondary w-100" onClick={onClose} disabled={loading}>
                  Close
                </button>
              </>
            ) : (
              <p style={{ color: "var(--bib-text-muted)" }}>No profile data.</p>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
