"""Expose the repository root as the comic_text_detector package.

The standalone checkout is named comic-text-detector, which is not a valid
module name, so the tests import the code through a package module.
"""

import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

if _REPO_ROOT.name == 'comic_text_detector':
    _parent = str(_REPO_ROOT.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
elif 'comic_text_detector' not in sys.modules:
    _pkg = types.ModuleType('comic_text_detector')
    _pkg.__path__ = [str(_REPO_ROOT)]
    sys.modules['comic_text_detector'] = _pkg
