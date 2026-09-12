/** Union-find (disjoint set) over integer indices 0..n-1, with path compression and union by rank. */
export class UnionFind {
  private parent: number[];
  private rank: number[];

  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x: number): number {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }

  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return;
    if (this.rank[ra] < this.rank[rb]) {
      this.parent[ra] = rb;
    } else if (this.rank[ra] > this.rank[rb]) {
      this.parent[rb] = ra;
    } else {
      this.parent[rb] = ra;
      this.rank[ra]++;
    }
  }

  /**
   * Assign a rank to each index by component, ordered by the first appearance of that
   * component among the given ids (0..n-1 in order). Components with more than one member
   * get rank 0, 1, 2, ... in order of first appearance; singleton components (size 1) get -1.
   */
  componentRank(ids: number[]): number[] {
    const n = ids.length;
    const roots = ids.map((i) => this.find(i));
    const size = new Map<number, number>();
    for (const r of roots) size.set(r, (size.get(r) ?? 0) + 1);

    const rankOf = new Map<number, number>();
    let nextRank = 0;
    const result = new Array<number>(n);
    for (let i = 0; i < n; i++) {
      const r = roots[i];
      if ((size.get(r) ?? 0) <= 1) {
        result[i] = -1;
        continue;
      }
      if (!rankOf.has(r)) {
        rankOf.set(r, nextRank++);
      }
      result[i] = rankOf.get(r)!;
    }
    return result;
  }
}
