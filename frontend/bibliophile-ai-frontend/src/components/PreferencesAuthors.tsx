import { useState, useEffect } from "react";

type PreferencesAuthorsProps = {
  token: string;
  initialSelectedAuthors?: string[];
  onSave: (authors: string[]) => void;
  onBack: () => void;
  loading?: boolean;
};

export default function PreferencesAuthors({
  token,
  initialSelectedAuthors = [],
  onSave,
  onBack,
  loading = false,
}: PreferencesAuthorsProps) {
  const [allAuthors, setAllAuthors] = useState<string[]>([]);
  const [selectedAuthors, setSelectedAuthors] = useState<string[]>(initialSelectedAuthors);

useEffect(() => {
  const controller = new AbortController();
  fetch("http://localhost:8000/popular-authors", {
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
  setSelectedAuthors((prevAuthors) =>
    prevAuthors.includes(author)
      ? prevAuthors.filter((a) => a !== author)
      : [...prevAuthors, author]
  );
};

  return (
    <div className="p-4">
      <h3>Select your favorite authors</h3>
      <div className="row">
        {allAuthors.map((author) => (
          <div key={author} className="col-6 col-md-4 mb-2">
            <button
              className={`btn w-100 ${selectedAuthors.includes(author) ? "btn-primary" : "btn-outline-primary"}`}
              onClick={() => toggleAuthor(author)}
              disabled={loading}
            >
              {author}
            </button>
          </div>
        ))}
      </div>
      <div className="mt-3">
        <button className="btn btn-secondary me-2" onClick={onBack} disabled={loading}>Back</button>
        <button className="btn btn-success" onClick={() => onSave(selectedAuthors)} disabled={selectedAuthors.length === 0 || loading}>
          {loading ? "Saving..." : "Next"}
        </button>
      </div>
    </div>
  );
}
