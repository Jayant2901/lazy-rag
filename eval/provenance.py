import subprocess
from datetime import datetime, timezone


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def provenance_metadata(model: str) -> dict:
    """Small metadata block recording when/against-what a result file was generated."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "model": model,
    }
