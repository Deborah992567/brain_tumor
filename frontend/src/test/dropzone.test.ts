import { describe, expect, it } from "vitest";
import { validateImageFile } from "../components/Dropzone";

function makeFile(name: string, sizeMb = 0.01): File {
  const bytes = new Uint8Array(Math.max(1, Math.floor(sizeMb * 1024 * 1024)));
  return new File([bytes], name, { type: "image/jpeg" });
}

describe("validateImageFile", () => {
  it("accepts supported raster formats", () => {
    expect(
      validateImageFile(makeFile("scan.jpg")),
    ).toBeNull();
    expect(
      validateImageFile(makeFile("scan.png")),
    ).toBeNull();
    expect(
      validateImageFile(makeFile("scan.tiff")),
    ).toBeNull();
  });

  it("rejects unsupported extensions", () => {
    expect(
      validateImageFile(makeFile("scan.gif")),
    ).toMatch(/Unsupported format/);
    expect(
      validateImageFile(makeFile("report.pdf")),
    ).toMatch(/Unsupported format/);
  });

  it("rejects files larger than 10 MB", () => {
    expect(
      validateImageFile(makeFile("huge.jpg", 20)),
    ).toMatch(/exceeds the 10 MB/);
  });
});