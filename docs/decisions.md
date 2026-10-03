# Decision Log

Record significant project decisions here after they are discussed and agreed.

Copy the template below for each decision.

## YYYY-MM-DD — Decision title

- **Status:** Proposed | Accepted | Superseded
- **Context:** Describe the problem or constraint that required a decision.
- **Decision:** State the agreed choice.
- **Consequences:** Record the important benefits, tradeoffs, and follow-up work.
- **Supersedes:** Reference an earlier decision when applicable.

## 2026-10-03 — Shared Next.js frontend

- **Status:** Accepted
- **Context:** The project needs a common frontend starting point and a public landing page. The backend currently exposes only a health endpoint.
- **Decision:** Place a TypeScript Next.js App Router application in `frontend/`.
- **Consequences:** Frontend lint and build checks added to CI.
- **Supersedes:** None.
