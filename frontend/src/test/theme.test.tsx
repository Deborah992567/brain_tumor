import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider, useTheme } from "../theme";

function Probe() {
  const { theme, preference, setPreference } = useTheme();
  return (
    <div>
      <span data-testid="theme">{theme}</span>
      <span data-testid="preference">{preference}</span>
      <button onClick={() => setPreference("dark")}>go-dark</button>
      <button onClick={() => setPreference("light")}>go-light</button>
    </div>
  );
}

describe("ThemeProvider", () => {
  it("defaults to system preference and applies the theme attribute", () => {
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    expect(screen.getByTestId("preference").textContent).toBe("system");
    expect(document.documentElement.getAttribute("data-theme")).toBe("light");
  });

  it("switches theme and persists the choice", async () => {
    const user = userEvent.setup();
    render(
      <ThemeProvider>
        <Probe />
      </ThemeProvider>,
    );
    await user.click(screen.getByText("go-dark"));
    expect(screen.getByTestId("theme").textContent).toBe("dark");
    expect(screen.getByTestId("preference").textContent).toBe("dark");
    expect(localStorage.getItem("btai-theme")).toBe("dark");
    expect(document.documentElement.getAttribute("data-theme")).toBe("dark");
  });
});