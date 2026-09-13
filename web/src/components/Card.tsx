import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
  /** Optional uppercase label rendered above the card body. */
  title?: string;
  /** Extra classes — the card keeps its own surface/border/padding. */
  className?: string;
  /** Drops the padding so a canvas or image can bleed to the rounded edge. */
  flush?: boolean;
}

/** Surface + border + padding. The one container every panel sits in. */
export function Card({ children, title, className, flush }: Props) {
  const classes = ["card", flush ? "card-flush" : "", className ?? ""]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={classes}>
      {title && <div className="card-title">{title}</div>}
      {children}
    </div>
  );
}
