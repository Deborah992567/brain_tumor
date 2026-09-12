import { HashRouter, Route, Routes } from "react-router-dom";
import { AppLayout } from "./components/Layout";
import { ToastProvider } from "./components/Toasts";
import { ThemeProvider } from "./theme";
import { Dashboard } from "./pages/Dashboard";
import { NewAnalysis } from "./pages/NewAnalysis";
import { Results } from "./pages/Results";
import { History } from "./pages/History";
import { Reports } from "./pages/Reports";
import { Models } from "./pages/Models";
import { Research } from "./pages/Research";
import { Settings } from "./pages/Settings";

export default function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <HashRouter>
          <Routes>
            <Route element={<AppLayout />}>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analysis/new" element={<NewAnalysis />} />
              <Route path="/analysis/:analysisId" element={<Results />} />
              <Route path="/history" element={<History />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/models" element={<Models />} />
              <Route path="/research" element={<Research />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<Dashboard />} />
            </Route>
          </Routes>
        </HashRouter>
      </ToastProvider>
    </ThemeProvider>
  );
}