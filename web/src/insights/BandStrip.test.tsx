// The strip is vertical beside the spectrogram on a desktop and HORIZONTAL
// under 560px, which is the phone and therefore the common case. The axis is
// read off the measured rect, so these tests fake the rect rather than a
// media query.

import { fireEvent, render, screen } from "@testing-library/react";
import { BandStrip } from "./BandStrip";
import { soloBand, useSoloError } from "../audio/soloStore";

vi.mock("../audio/soloStore", () => ({
  soloBand: vi.fn(),
  useSoloError: vi.fn(() => null),
}));

/** 4 bands, edges every 100 Hz: band i is [100i, 100i + 200]. */
const BANDS = 4;
const EDGES = new Float64Array([0, 100, 200, 300, 400, 500]);

function shape(rect: { width: number; height: number }) {
  const track = screen.getByRole("slider");
  vi.spyOn(track, "getBoundingClientRect").mockReturnValue({
    left: 0, top: 0, right: rect.width, bottom: rect.height,
    width: rect.width, height: rect.height, x: 0, y: 0, toJSON: () => {},
  } as DOMRect);
  // setPointerCapture isn't in jsdom.
  (track as unknown as { setPointerCapture: (id: number) => void }).setPointerCapture = () => {};
  return track;
}

beforeEach(() => {
  vi.mocked(soloBand).mockReset();
  vi.mocked(useSoloError).mockReturnValue(null);
});

test("vertical geometry: low frequency is at the bottom", () => {
  render(<BandStrip bandEdgesHz={EDGES} bands={BANDS} band={null} />);
  const track = shape({ width: 18, height: 400 });

  // 10 px from the bottom of a 400 px strip = band 0, the lowest.
  fireEvent.pointerDown(track, { clientX: 9, clientY: 390, pointerId: 1 });
  expect(soloBand).toHaveBeenLastCalledWith(0, 200);

  // Drag to the top: bands 0..3, i.e. 0 Hz to the top edge.
  fireEvent.pointerMove(track, { clientX: 9, clientY: 5, pointerId: 1 });
  expect(soloBand).toHaveBeenLastCalledWith(0, 500);
});

test("horizontal geometry: low frequency is at the left, and a drag maps across", () => {
  render(<BandStrip bandEdgesHz={EDGES} bands={BANDS} band={null} />);
  const track = shape({ width: 400, height: 18 });

  // 10 px from the LEFT of a 400 px strip = band 0.
  fireEvent.pointerDown(track, { clientX: 10, clientY: 9, pointerId: 1 });
  expect(soloBand).toHaveBeenLastCalledWith(0, 200);

  // 250/400 = band 2.
  fireEvent.pointerMove(track, { clientX: 250, clientY: 9, pointerId: 1 });
  expect(soloBand).toHaveBeenLastCalledWith(0, 400);

  // Past the right edge clamps to the last band rather than overflowing.
  fireEvent.pointerMove(track, { clientX: 999, clientY: 9, pointerId: 1 });
  expect(soloBand).toHaveBeenLastCalledWith(0, 500);
});

test("the horizontal selection is drawn with left/width, the vertical with bottom/height", () => {
  const view = render(<BandStrip bandEdgesHz={EDGES} bands={BANDS} band={[0, 200]} />);
  const track = shape({ width: 400, height: 18 });

  fireEvent.pointerDown(track, { clientX: 10, clientY: 9, pointerId: 1 });
  const horizontal = view.container.querySelector(".band-strip-selection") as HTMLElement;
  expect(horizontal.style.left).toBe("0%");
  expect(horizontal.style.width).toBe("25%");
  expect(horizontal.style.height).toBe("");
});

test("arrow keys move the band, and Shift moves it coarsely", () => {
  render(<BandStrip bandEdgesHz={EDGES} bands={16} band={null} />);
  const track = shape({ width: 18, height: 400 });

  fireEvent.keyDown(track, { key: "ArrowUp" });
  expect(soloBand).toHaveBeenLastCalledWith(EDGES[1], EDGES[3]); // band 1
  expect(track).toHaveAttribute("aria-valuenow", "1");

  fireEvent.keyDown(track, { key: "ArrowRight" });
  expect(track).toHaveAttribute("aria-valuenow", "2");

  fireEvent.keyDown(track, { key: "ArrowDown" });
  expect(track).toHaveAttribute("aria-valuenow", "1");

  fireEvent.keyDown(track, { key: "ArrowLeft" });
  expect(track).toHaveAttribute("aria-valuenow", "0");
  // Already at the floor: clamped, not negative.
  fireEvent.keyDown(track, { key: "ArrowLeft" });
  expect(track).toHaveAttribute("aria-valuenow", "0");

  fireEvent.keyDown(track, { key: "ArrowUp", shiftKey: true });
  expect(track).toHaveAttribute("aria-valuenow", "8");
});

test("a failed attach replaces the label, in the danger colour", () => {
  vi.mocked(useSoloError).mockReturnValue("Couldn't enable band solo.");
  const view = render(<BandStrip bandEdgesHz={EDGES} bands={BANDS} band={[0, 200]} />);

  const label = view.container.querySelector(".band-strip-label") as HTMLElement;
  expect(label.textContent).toBe("Couldn't enable band solo.");
  expect(label.className).toContain("band-strip-label-error");
});

test("with no error the label shows the soloed range", () => {
  const view = render(<BandStrip bandEdgesHz={EDGES} bands={BANDS} band={[240, 1200]} />);
  const label = view.container.querySelector(".band-strip-label") as HTMLElement;
  expect(label.textContent).toBe("240 – 1.2k Hz");
  expect(label.className).not.toContain("band-strip-label-error");
});
