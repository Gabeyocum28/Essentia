interface Props {
  width?: number | string;
  height?: number | string;
  radius?: number | string;
  className?: string;
}

/** A shimmering placeholder block. Decorative, so it is hidden from a11y. */
export function Skeleton({ width, height = 14, radius, className }: Props) {
  return (
    <div
      className={className ? `skeleton ${className}` : "skeleton"}
      aria-hidden="true"
      style={{ width, height, borderRadius: radius }}
    />
  );
}

/** The track-row shape, repeated — used while a list of tracks is loading. */
export function SkeletonTrackList({ rows = 5 }: { rows?: number }) {
  return (
    <div className="skeleton-list" aria-hidden="true">
      {Array.from({ length: rows }, (_, i) => (
        <div className="skeleton-row" key={i}>
          <Skeleton width={34} height={34} radius={999} />
          <Skeleton width={44} height={44} radius={8} />
          <div className="skeleton-row-lines">
            <Skeleton width={`${58 - (i % 3) * 9}%`} height={12} />
            <Skeleton width={`${36 - (i % 3) * 6}%`} height={10} />
          </div>
        </div>
      ))}
    </div>
  );
}
