"""Test the Renderer class."""

import unittest
from pathlib import Path

from image_renderer.Renderer import Renderer


class TestRenderer(unittest.TestCase):
    """Test the Renderer class."""

    def test_render(self):
        self.renderer = Renderer()
        self.renderer.render_object(
            Path("sample_data/sample_from_abc.stl"),
            Path("sample_output"),
            three_views=True,
        )


if __name__ == "__main__":
    unittest.main()
