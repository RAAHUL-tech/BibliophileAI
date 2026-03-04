# BibliophileAI Frontend

React (Vite + TypeScript) single-page application for the BibliophileAI book recommendation and reading experience. It talks to the backend APIs via the Kubernetes Ingress (e.g. `http://localhost:8080`) for auth, recommendations, and search.

## Functionality

- **Auth**: Login (username/password), Google OAuth, and registration. JWT and session ID are stored in sessionStorage and sent with API requests.
- **Onboarding**: After registration, new users go through genre selection, author selection, and demographics (pincode, age); data is sent to the user service and used for recommendations and user embeddings.
- **Homepage**: Fetches combined recommendations from `/api/v1/recommend/combined` and displays them in Netflix-style category rows (Content-Based, Collaborative Filtering, Trending Now, For You (LinUCB), Top Picks, etc.). Search bar opens a search overlay; clicking a book opens a book overlay or the reader.
- **Search**: Search bar (e.g. on Enter) sends a POST to `/api/v1/search` and shows results in `SearchOverlay` (horizontal slider of book cards).
- **Book overlay**: Shows book details and bookmark button; link to open the EPUB reader (`BookView`).
- **Book view (reader)**: Loads EPUB from S3 (URL from backend), renders with `react-reader`; supports rating/review and sends page-turn events to the backend for clickstream and recommendations.
- **Profile**: User profile, preferences (genre/author) with inline edit, bookmarks carousel. Theme toggle (dark/light) is available in the navbar.
- **Theme**: Dark/light theme persisted in localStorage and applied via `data-theme` on the document; `ThemeContext` provides theme state and toggle.

## Implementation in this project

- **Stack**: Vite, React 18, TypeScript, React Router, Bootstrap, react-reader for EPUBs, Google OAuth (`@react-oauth/google`).
- **Entry**: `main.tsx` mounts the app with `BrowserRouter`, `GoogleOAuthProvider`, and `ThemeProvider`; `App.tsx` holds auth state and routes (login, register, onboarding, homepage).
- **API base**: Requests use a configurable base URL (e.g. `http://localhost:8080` for dev with Ingress port-forward); auth header `Bearer <token>` for protected endpoints.
- **Key components**: `Homepage` (recommendations + search + profile dropdown), `SearchOverlay`, `BookOverlay`, `BookView`, `Profile`, `UserOnboarding`, `PreferencesGenre`, `PreferencesAuthors`, `DemographicForm`, `UpdateGenrePreferences`, `UpdateAuthorPreferences`, `Login`, `Register`.

## Running

- **Dev**: `npm install && npm run dev`; Vite runs on port 5173. Point API to Ingress (e.g. 8080) via env or config.
- **Build**: `npm run build`; output in `dist/` for static hosting or containerization.
