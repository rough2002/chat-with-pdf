HISTORY_TOKEN_BUDGET = 15000

def count_tokens(text: str) -> int:
    # rough estimate: ~4 characters per token
    return len(text) // 4 + 1

def trim_history(messages: list[dict], budget: int = HISTORY_TOKEN_BUDGET) -> list[dict]:
    # make sure they're oldest -> newest
    messages = sorted(messages, key=lambda m: m["id"])

    recent_history = []
    used_tokens = 0

    for msg in reversed(messages):               # newest first
        tokens = count_tokens(msg["message"])
        if used_tokens + tokens > budget:
            break                                # stop at the first one that doesn't fit
        recent_history.append(msg)
        used_tokens += tokens

    recent_history.reverse()                     # back to oldest -> newest
    return recent_history