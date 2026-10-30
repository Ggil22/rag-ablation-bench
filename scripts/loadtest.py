"""Load test for the /ask endpoint: latency percentiles under concurrency.

    uv run python scripts/loadtest.py --url http://localhost:8080 --concurrency 4 --repeat 2

Questions come from an evaluation set, so every request is realistic. Note that LLM
responses are cached on disk: run once to warm the cache and measure retrieval and serving
overhead, or clear .cache/llm to include real generation latency.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from pathlib import Path

import httpx
import numpy as np
import yaml


async def worker(client, url, queue, results):
    while True:
        try:
            question = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        t0 = time.perf_counter()
        try:
            r = await client.post(f"{url}/ask", json={"question": question}, timeout=180)
            ok = r.status_code == 200
            body = r.json() if ok else {}
        except httpx.HTTPError:
            ok, body = False, {}
        results.append({"ok": ok, "latency_s": time.perf_counter() - t0, **body})


async def main(args) -> None:
    questions = [q["question"] for q in yaml.safe_load(args.questions.read_text(encoding="utf-8"))]
    queue: asyncio.Queue[str] = asyncio.Queue()
    for _ in range(args.repeat):
        for q in questions:
            queue.put_nowait(q)
    results: list[dict] = []
    t0 = time.perf_counter()
    async with httpx.AsyncClient() as client:
        await asyncio.gather(
            *(worker(client, args.url, queue, results) for _ in range(args.concurrency))
        )
    wall = time.perf_counter() - t0

    ok = [r for r in results if r["ok"]]
    lat = np.array([r["latency_s"] for r in ok])
    print(f"requests: {len(results)}  ok: {len(ok)}  concurrency: {args.concurrency}")
    print(f"throughput: {len(ok) / wall:.2f} req/s")
    if ok:
        p50, p95, p99 = np.percentile(lat, [50, 95, 99])
        print(f"latency  p50 {p50:.2f} s  p95 {p95:.2f} s  p99 {p99:.2f} s  max {lat.max():.2f} s")
        print(
            f"retrieval median {statistics.median(r['retrieval_ms'] for r in ok):.0f} ms, "
            f"generation median {statistics.median(r['generation_ms'] for r in ok):.0f} ms"
        )
        costs = [r["cost_usd"] for r in ok if r.get("cost_usd") is not None]
        if costs:
            print(f"cost: ${1000 * statistics.mean(costs):.3f} per 1,000 queries")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--questions", type=Path, default=Path("evalsets/boe/seed.yaml"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=1)
    asyncio.run(main(parser.parse_args()))
