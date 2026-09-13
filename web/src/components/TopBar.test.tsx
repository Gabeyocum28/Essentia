import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { TopBar } from "./TopBar";
import { Card } from "./Card";
import { Skeleton, SkeletonTrackList } from "./Skeleton";

function renderBar(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <TopBar />
    </MemoryRouter>,
  );
}

test("top bar shows the brand and no back link at home", () => {
  const { container } = renderBar("/");
  expect(screen.getByRole("link", { name: "Essentia" })).toBeInTheDocument();
  expect(screen.getByText("Search")).toBeInTheDocument();
  expect(screen.queryByRole("link", { name: "← Back" })).not.toBeInTheDocument();
  expect(container.querySelector(".top-bar-mark")).not.toBeNull();
});

test("top bar labels the section and offers a back link off home", () => {
  renderBar("/insights/42/energy");
  expect(screen.getByText("Insights")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "← Back" })).toBeInTheDocument();
});

test("card renders a title and its children on the card surface", () => {
  const { container } = render(
    <Card title="Proof">
      <p>body</p>
    </Card>,
  );
  expect(screen.getByText("Proof")).toBeInTheDocument();
  expect(screen.getByText("body")).toBeInTheDocument();
  expect(container.querySelector(".card")).not.toBeNull();
});

test("skeletons are decorative placeholders", () => {
  const { container } = render(
    <div>
      <Skeleton width={120} height={12} />
      <SkeletonTrackList rows={3} />
    </div>,
  );
  const blocks = container.querySelectorAll(".skeleton");
  // one standalone + three rows × four blocks each
  expect(blocks.length).toBe(13);
  for (const block of blocks) {
    expect(block.closest("[aria-hidden='true']")).not.toBeNull();
  }
  expect(container.querySelectorAll(".skeleton-row").length).toBe(3);
});
