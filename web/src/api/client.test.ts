import { api, ApiError, decodeCoords8, previewUrl } from "./client";

function mockFetch(status: number, body: unknown) {
  globalThis.fetch = vi.fn(async () => new Response(JSON.stringify(body), { status,
    headers: { "content-type": "application/json" } })) as unknown as typeof fetch;
}

test("search hits /api/search with the query encoded", async () => {
  mockFetch(200, { results: [] });
  await api.search("miles davis");
  expect(fetch).toHaveBeenCalledWith("/api/search?q=miles+davis", expect.anything());
});

test("seed posts JSON and returns status", async () => {
  mockFetch(200, { track_id: "1", status: "ready" });
  const r = await api.seed("1");
  expect(r.status).toBe("ready");
  const [, init] = (fetch as any).mock.calls[0];
  expect(init.method).toBe("POST");
  expect(JSON.parse(init.body)).toEqual({ track_id: "1" });
});

test("non-2xx becomes ApiError with the server detail", async () => {
  mockFetch(502, { detail: "analysis failed" });
  await expect(api.seed("1")).rejects.toMatchObject({ status: 502, detail: "analysis failed" });
  await expect(api.seed("1")).rejects.toBeInstanceOf(ApiError);
});

test("decodeCoords8 splits little-endian float32 into rows of 8", () => {
  const f = new Float32Array(16).map((_, i) => i);
  const b64 = btoa(String.fromCharCode(...new Uint8Array(f.buffer)));
  const rows = decodeCoords8(b64, 2);
  expect(rows.length).toBe(2);
  expect(Array.from(rows[1])).toEqual([8, 9, 10, 11, 12, 13, 14, 15]);
});

test("previewUrl points at the API redirect", () => {
  expect(previewUrl("42")).toBe("/api/preview/42");
});

test("vizHubs adds track_id when given, omits it otherwise", async () => {
  mockFetch(200, { hubs: [], central: [], isolated: [], expected_k: 8 });
  await api.vizHubs("42");
  expect(fetch).toHaveBeenCalledWith("/api/viz/hubs?track_id=42", expect.anything());

  mockFetch(200, { hubs: [], central: [], isolated: [], expected_k: 8 });
  await api.vizHubs();
  expect(fetch).toHaveBeenCalledWith("/api/viz/hubs?", expect.anything());
});

test("vizTour adds track_id when given, omits it otherwise", async () => {
  mockFetch(200, { ids: [], coords8: "", variance: [] });
  await api.vizTour("42");
  expect(fetch).toHaveBeenCalledWith("/api/viz/tour?track_id=42", expect.anything());

  mockFetch(200, { ids: [], coords8: "", variance: [] });
  await api.vizTour();
  expect(fetch).toHaveBeenCalledWith("/api/viz/tour?", expect.anything());
});

test("vizMst adds track_id when given, omits it otherwise", async () => {
  mockFetch(200, { ids: [], edges: [] });
  await api.vizMst("42");
  expect(fetch).toHaveBeenCalledWith("/api/viz/mst?track_id=42", expect.anything());

  mockFetch(200, { ids: [], edges: [] });
  await api.vizMst();
  expect(fetch).toHaveBeenCalledWith("/api/viz/mst?", expect.anything());
});

test("vizExtremes adds track_id when given, omits it otherwise", async () => {
  mockFetch(200, { pc: 1, variance_pct: 10, low: [], high: [] });
  await api.vizExtremes(1, 4, "42");
  expect(fetch).toHaveBeenCalledWith("/api/viz/extremes?pc=1&limit=4&track_id=42", expect.anything());

  mockFetch(200, { pc: 1, variance_pct: 10, low: [], high: [] });
  await api.vizExtremes(1, 4);
  expect(fetch).toHaveBeenCalledWith("/api/viz/extremes?pc=1&limit=4", expect.anything());
});

test("the global viz endpoints send the rec ids as one comma-separated param", async () => {
  // A `surprise` rec can sit outside the seed's subset; the server only keeps
  // a row for it if it is named here.
  mockFetch(200, { hubs: [], central: [], isolated: [], expected_k: 8 });
  await api.vizHubs("42", ["a", "b"]);
  expect(fetch).toHaveBeenCalledWith("/api/viz/hubs?track_id=42&recs=a%2Cb", expect.anything());

  mockFetch(200, { ids: [], coords8: "", variance: [] });
  await api.vizTour("42", ["a", "b"]);
  expect(fetch).toHaveBeenCalledWith("/api/viz/tour?track_id=42&recs=a%2Cb", expect.anything());

  mockFetch(200, { ids: [], edges: [] });
  await api.vizMst("42", ["a"]);
  expect(fetch).toHaveBeenCalledWith("/api/viz/mst?track_id=42&recs=a", expect.anything());

  mockFetch(200, { pc: 1, variance_pct: 10, low: [], high: [] });
  await api.vizExtremes(1, 4, "42", ["a"]);
  expect(fetch).toHaveBeenCalledWith(
    "/api/viz/extremes?pc=1&limit=4&track_id=42&recs=a",
    expect.anything(),
  );
});

test("an empty rec list omits the param rather than sending recs=", async () => {
  mockFetch(200, { ids: [], edges: [] });
  await api.vizMst("42", []);
  expect(fetch).toHaveBeenCalledWith("/api/viz/mst?track_id=42", expect.anything());
});

test("recommend adds feel when given, omits it otherwise", async () => {
  mockFetch(200, { seed_track_id: "1", axis: "energy", results: [] });
  await api.recommend("1", "energy", 10, 0.5);
  expect(fetch).toHaveBeenCalledWith("/api/recommend?track_id=1&axis=energy&limit=10&feel=0.5", expect.anything());

  mockFetch(200, { seed_track_id: "1", axis: "energy", results: [] });
  await api.recommend("1", "energy");
  expect(fetch).toHaveBeenCalledWith("/api/recommend?track_id=1&axis=energy&limit=10", expect.anything());
});

test("vizMap adds feel when given, omits it otherwise", async () => {
  mockFetch(200, { points: { ids: [], x: [], y: [], tracks: [] }, seed: {}, recs: [], axis: {} });
  await api.vizMap("1", "energy", 10, "on", 0.5);
  expect(fetch).toHaveBeenCalledWith(
    "/api/viz/map?track_id=1&axis=energy&limit=10&correction=on&feel=0.5",
    expect.anything(),
  );

  mockFetch(200, { points: { ids: [], x: [], y: [], tracks: [] }, seed: {}, recs: [], axis: {} });
  await api.vizMap("1", "energy");
  expect(fetch).toHaveBeenCalledWith("/api/viz/map?track_id=1&axis=energy&limit=10", expect.anything());
});
