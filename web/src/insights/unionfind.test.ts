import { describe, expect, test } from "vitest";
import { UnionFind } from "./unionfind";

describe("UnionFind", () => {
  test("union merges components so find agrees", () => {
    const uf = new UnionFind(4);
    expect(uf.find(0)).not.toBe(uf.find(1));
    uf.union(0, 1);
    expect(uf.find(0)).toBe(uf.find(1));
    expect(uf.find(2)).not.toBe(uf.find(0));
  });

  test("componentRank ranks by first appearance among non-singleton components; singletons are -1", () => {
    const uf = new UnionFind(4);
    uf.union(0, 1);
    const ranks = uf.componentRank([0, 1, 2, 3]);
    expect(ranks).toEqual([0, 0, -1, -1]);
  });

  test("componentRank assigns ranks in order of first appearance across multiple components", () => {
    const uf = new UnionFind(6);
    uf.union(2, 3);
    uf.union(4, 5);
    const ranks = uf.componentRank([0, 1, 2, 3, 4, 5]);
    expect(ranks).toEqual([-1, -1, 0, 0, 1, 1]);
  });
});
