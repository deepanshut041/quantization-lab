"""Declare future entry points without simulating their completion."""

PENDING_ACTIONS = ("prepare-data","encode","build-index","evaluate","search",)


def pending_result(action: str) -> dict:
    if action not in PENDING_ACTIONS:
        raise ValueError(f"Unknown pending action: {action}")
    return {
        "project": "compressed-semantic-search",
        "action": action,
        "status": "not_implemented",
        "message": "Corpus preparation, encoding, index construction, evaluation and search remain pending. No model or output was created.",
    }
