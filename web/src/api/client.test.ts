import { describe, expect, it } from "vitest";
import { friendlyMessage } from "./client";

describe("friendlyMessage", () => {
  it.each([400, 413, 415, 503, null])("gives a plain-language message for %s", (status) => {
    const msg = friendlyMessage(status);
    expect(msg.length).toBeGreaterThan(20);
    expect(msg).not.toMatch(/\b(traceback|exception|HTTP)\b/i);
  });
  it("uses the server's text for 422", () => {
    expect(friendlyMessage(422, "Part B is not loaded.")).toBe("Part B is not loaded.");
  });
});
