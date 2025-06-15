import requests
from io import BytesIO
from PIL import Image
import numpy as np


class Crawler:

    @staticmethod
    def fetch_image_from_url(url: str) -> np.ndarray:
        """Downloads an image from a URL and returns it as a numpy array."""
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return np.array(image)