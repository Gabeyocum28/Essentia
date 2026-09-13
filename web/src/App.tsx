import { BrowserRouter, Route, Routes } from "react-router-dom";
import { PlayerProvider } from "./player/usePlayer";
import { Player } from "./player/Player";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { TopBar } from "./components/TopBar";
import { Search } from "./screens/Search";
import { Seed } from "./screens/Seed";
import { Recommendations } from "./screens/Recommendations";
import { Insights } from "./screens/Insights";

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
