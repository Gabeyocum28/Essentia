interface Props {
  url: string | null | undefined;
  size: number;
}

export function Artwork({ url, size }: Props) {
  return (
    <img
      className="art"
      src={url || undefined}
      width={size}
      height={size}
      style={{ width: size, height: size, flexShrink: 0 }}
      alt=""
    />
  );
}
