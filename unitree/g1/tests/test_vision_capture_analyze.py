"""vision_capture analyze: brightness verdict on saved photos (no camera needed)."""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile
import threading
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_class(name):
    tree = ast.parse((ROOT / "device.py").read_text())
    class_node = next(node for node in tree.body
                      if isinstance(node, ast.ClassDef) and node.name == name)
    namespace = globals().copy()
    namespace.update({"_VISION_FIRST_FRAME_TIMEOUT_S": 5.0,
                      "_VISION_MAX_FRAME_AGE_S": 3.0,
                      "_vision_acp_notify": lambda *a: None,
                      "Path": Path, "threading": threading})
    exec(compile(ast.Module(body=[class_node], type_ignores=[]),
                 str(ROOT / "device.py"), "exec"), namespace)
    return namespace[name]


def _plugin(tmp):
    cls = _load_class("VisionCapturePlugin")
    plugin = cls.__new__(cls)
    plugin._output_dir = Path(tmp)
    plugin._dark_threshold = 60.0
    plugin._lit_threshold = 110.0
    plugin._last_photo_path = None
    (Path(tmp) / "photos").mkdir()
    return plugin


def _write(tmp, name, value):
    img = np.full((120, 160, 3), value, dtype=np.uint8)
    path = Path(tmp) / "photos" / name
    cv2.imwrite(str(path), img)
    return path


class AnalyzeTests(unittest.TestCase):
    def test_dark_and_lit_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = _plugin(tmp)
            dark = _write(tmp, "IMG_1.jpg", 20)
            lit = _write(tmp, "IMG_2.jpg", 200)
            self.assertEqual("dark", plugin.dispatch("analyze", {"image_path": str(dark)})["verdict"])
            self.assertEqual("lit", plugin.dispatch("analyze", {"image_path": str(lit)})["verdict"])

    def test_defaults_to_latest_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = _plugin(tmp)
            _write(tmp, "IMG_1.jpg", 20)
            _write(tmp, "IMG_2.jpg", 200)
            result = plugin.dispatch("analyze", {})
            self.assertTrue(result["ok"])
            self.assertTrue(result["file_path"].endswith("IMG_2.jpg"))

    def test_rejects_path_outside_photos_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = _plugin(tmp)
            outside = Path(tmp) / "x.jpg"
            cv2.imwrite(str(outside), np.zeros((10, 10, 3), dtype=np.uint8))
            result = plugin.dispatch("analyze", {"image_path": str(outside)})
            self.assertFalse(result["ok"])
            self.assertEqual("INVALID_IMAGE", result["code"])

    def test_schema_lists_analyze(self):
        with tempfile.TemporaryDirectory() as tmp:
            plugin = _plugin(tmp)
            plugin._fps, plugin._max_duration_s = 15, 30
            schema = plugin.get_tool()["inputSchema"]
            self.assertIn("analyze", schema["properties"]["action"]["enum"])
            self.assertIn("analyze", schema["x-action-params"])


if __name__ == "__main__":
    unittest.main()
