import { act, render, screen } from "@testing-library/react";
import { WhySimilar, hz } from "./WhySimilar";
import { api } from "../api/client";
import { soloBand, soloOff, useSoloBand, useSoloError } from "../audio/soloStore";

vi.mock("../audio/soloStore", () => ({
  soloBand: vi.fn(),
  soloOff: vi.fn(),
  useSoloBand: vi.fn(() => null),
  useSoloError: vi.fn(() => null),
}));

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    api: {
      ...actual.api,
      vizAttribute: vi.fn(),
    },
  };
});

beforeEach(() => {
  vi.useFakeTimers();
  vi.mocked(api.vizAttribute).mockReset();
  vi.mocked(soloBand).mockReset();
  vi.mocked(soloOff).mockReset();
  vi.mocked(useSoloBand).mockReturnValue(null);
  vi.mocked(useSoloError).mockReturnValue(null);
});

afterEach(() => {
  vi.useRealTimers();
});

test("polls while pending, renders bars once ready, then stops polling", async () => {
  vi.mocked(api.vizAttribute)
    .mockResolvedValueOnce({ status: "pending" })
    .mockResolvedValueOnce({ status: "pending" })
    .mockResolvedValueOnce({
      status: "ready",
      base: 0.8234,
      bands: [
        { lo_hz: 20, hi_hz: 240, delta: 0.1 },
        { lo_hz: 240, hi_hz: 1200, delta: 0.4 },
      ],
    });

  render(<WhySimilar seedId="seed" recId="rec" />);

  await act(async () => {
    await Promise.resolve();
  });
  expect(api.vizAttribute).toHaveBeenCalledTimes(1);

  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500);
  });
  expect(api.vizAttribute).toHaveBeenCalledTimes(2);

  await act(async () => {
    await vi.advanceTimersByTimeAsync(1500);
  });
  expect(api.vizAttribute).toHaveBeenCalledTimes(3);

  expect(screen.getByText("20 / 240")).toBeInTheDocument();
  expect(screen.getByText("240 / 1.2k")).toBeInTheDocument();
  expect(screen.getByText("base cos 0.8234")).toBeInTheDocument();

  // Polling must have stopped: advancing well past another interval shouldn't add more calls.
  await act(async () => {
    await vi.advanceTimersByTimeAsync(10_000);
  });
  expect(api.vizAttribute).toHaveBeenCalledTimes(3);
});

test("shows the error message when the worker fails", async () => {
  vi.mocked(api.vizAttribute).mockResolvedValueOnce({ status: "failed", error: "boom" });
  render(<WhySimilar seedId="seed" recId="rec" />);

  await act(async () => {
    await Promise.resolve();
  });
  expect(screen.getByText("boom")).toBeInTheDocument();
});

test("clicking a bar solos that band; clicking the soloed one turns solo off", async () => {
  vi.mocked(api.vizAttribute).mockResolvedValueOnce({
    status: "ready",
    base: 0.5,
    bands: [
      { lo_hz: 20, hi_hz: 240, delta: 0.1 },
      { lo_hz: 240, hi_hz: 1200, delta: 0.4 },
    ],
  });
  const view = render(<WhySimilar seedId="seed" recId="rec" />);
  await act(async () => {
    await Promise.resolve();
  });

  await act(async () => {
    screen.getByRole("button", { name: "Solo 240 to 1.2k Hz" }).click();
  });
  expect(soloBand).toHaveBeenCalledWith(240, 1200);

  // With that band soloed, the same bar is the "off" switch.
  vi.mocked(useSoloBand).mockReturnValue([240, 1200]);
  view.rerender(<WhySimilar seedId="seed" recId="rec" />);
  await act(async () => {
    screen.getByRole("button", { name: "Solo 240 to 1.2k Hz" }).click();
  });
  expect(soloOff).toHaveBeenCalled();
});

test("a failed band-solo attach is reported under the bars", async () => {
  vi.mocked(api.vizAttribute).mockResolvedValue({
    status: "ready",
    bands: [{ lo_hz: 20, hi_hz: 240, delta: 0.1 }],
  });
  vi.mocked(useSoloError).mockReturnValue("Couldn't enable band solo.");

  render(<WhySimilar seedId="seed" recId="rec" />);
  await act(async () => {
    await Promise.resolve();
  });

  // The bars are still there -- the attribution worked, only the audio didn't.
  expect(screen.getByText("20 / 240")).toBeInTheDocument();
  expect(screen.getByText("Couldn't enable band solo.")).toBeInTheDocument();
});

describe("hz", () => {
  test("formats sub-kHz values as plain integers", () => {
    expect(hz(240)).toBe("240");
    expect(hz(20)).toBe("20");
    expect(hz(999)).toBe("999");
  });

  test("formats kHz-and-above values with one decimal", () => {
    expect(hz(1200)).toBe("1.2k");
    expect(hz(1000)).toBe("1.0k");
    expect(hz(16000)).toBe("16.0k");
  });
});
