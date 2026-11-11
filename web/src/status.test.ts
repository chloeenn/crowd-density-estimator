import { describe, expect, it } from "vitest";
import { approxCount, rangeFor } from "./status";

const T = { low: 50, medium: 150, high: 400, critical: 800 };

describe("approxCount", () => {
  it.each([
    [0, "0"],
    [7, "7"],
    [541, "540"],
    [1234, "1,200"],
    [2987, "3,000"],
  ])("rounds %s to %s", (n, out) => expect(approxCount(n)).toBe(out));
});

describe("rangeFor", () => {
  it("covers every level without gaps", () => {
    expect(rangeFor("Low", T)).toBe("Under 150");
    expect(rangeFor("Medium", T)).toBe("150 – 399");
    expect(rangeFor("High", T)).toBe("400 – 799");
    expect(rangeFor("Critical", T)).toBe("800+");
  });
  it("shows a dash before thresholds load", () => expect(rangeFor("Low")).toBe("—"));
});
