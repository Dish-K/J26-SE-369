"""Print the model ids this key can call (never the key).

    python -m backend.app.probing.ai.list_models
"""

import asyncio
import sys

from backend.app.probing.ai.errors import LLMError
from backend.app.probing.ai.gemini import list_models


def main() -> int:
    try:
        models = asyncio.run(list_models())
    except LLMError as exc:
        print(f"could not list models: {exc}", file=sys.stderr)
        return 1
    for m in sorted(models, key=lambda m: m["id"]):
        print(f"{m['id']:<45} in={m['input_token_limit']} out={m['output_token_limit']}  {m['display_name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
