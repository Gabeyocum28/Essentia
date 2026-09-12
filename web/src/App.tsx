import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { PlayerProvider } from "./player/usePlayer";
import { Player } from "./player/Player";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { Search } from "./screens/Search";
import { Seed } from "./screens/Seed";
import { Recommendations } from "./screens/Recommendations";
import { Insights } from "./screens/Insights";

function TopBar() {
  const location = useLocation();
  return (
    <div className="top-bar">
      <Link to="/" className="top-bar-title">
        Essentia
      </Link>
      {location.pathname !== "/" && (
        <Link to="/" className="top-bar-back">
          ← Back
        </Link>
      )}
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <PlayerProvider>
        <TopBar />
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Search />} />
            <Route path="/seed/:id" element={<Seed />} />
            <Route path="/recs/:id/:axis" element={<Recommendations />} />
            <Route path="/insights/:id/:axis" element={<Insights />} />
          </Routes>
        </ErrorBoundary>
        <Player />
      </PlayerProvider>
    </BrowserRouter>
  );
}

export default App;
