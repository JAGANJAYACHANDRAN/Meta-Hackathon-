# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Pytest: ensure `server` and root models resolve without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
