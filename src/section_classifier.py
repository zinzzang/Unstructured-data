import yaml
from pathlib import Path

_MAP = None


def classify(title):
    global _MAP
    if _MAP is None:
        _MAP = yaml.safe_load((Path(__file__).resolve().parent.parent / "config" / "section_mapping.yaml").read_text())
    for area, kws in _MAP.items():
        if any(k in title for k in kws):
            return area
    return "UNMAPPED"
