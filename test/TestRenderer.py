"""Test the Renderer class."""

import unittest
from image_renderer.Renderer import Renderer


class TestRenderer(unittest.TestCase):
    """Test the Renderer class."""

    def test_render(self):
        self.renderer = Renderer()
        self.renderer.render_object(
            "sample_data/sample_from_abc.stl", "sample_output", three_views=True
        )


if __name__ == "__main__":
    unittest.main()
