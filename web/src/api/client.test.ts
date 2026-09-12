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
