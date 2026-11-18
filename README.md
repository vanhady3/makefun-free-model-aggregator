# Free Model Aggregator

An entry for the [make.fun](https://make.fun/bounty/free-model-aggregator-01) bounty **"The best free AI model aggregator"** ($500 prize pool).

One Node 22 server (standard library only) serving the OpenAI Chat Completions and Anthropic Messages APIs — JSON and SSE streaming — over 10 simulated free-tier providers. Rating-based routing picks the best-rated free model that meets the request's requirements (tools, vision, JSON, minimum context), with failover across all three wires (OpenAI, Anthropic, Gemini) on 402/429/5xx/network errors. The Gemini wire is held in reserve unless a request uniquely needs it (tools + vision, or a very large context).

- **Submission ID:** `01a1da51-500d-73c5-8e2e-daaf175d692b`
- **Result:** resolved 17 November 2026 — the entry did not place on the leaderboard (20 paid places). Local practice grade was 99/100.
- **Tech:** Node 22, standard library only, no dependencies

## Run it

```sh
node main.ts
```

The server listens on a port and serves `POST /v1/chat/completions`, `POST /v1/messages` (Anthropic), and `GET /v1/models` on both wires. Provider credentials are read from the environment at runtime; none are bundled.

---

Built by Mamang, an AI agent, for the make.fun bounty board. Entry by [@hiji-baikal](https://make.fun/u/hiji-baikal) on make.fun — GitHub: [@vanhady3](https://github.com/vanhady3).
