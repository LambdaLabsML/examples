from pathlib import Path


def find_checkpoint(root: Path, filename: str = "last.ckpt") -> Path:
    last_checkpoints = sorted(list(root.rglob(filename)))
    assert len(last_checkpoints) == 1, "Error: more than one checkpoint found."
    return last_checkpoints[0]
