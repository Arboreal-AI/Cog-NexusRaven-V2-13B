# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from transformers import pipeline

MODEL_NAME = "NexusRaven-V2-13B"
MODEL_CACHE = "cache"

EXAMPLE_PROMPT = '''
Function:
def get_weather_data(coordinates):
    """
    Fetches weather data from the Open-Meteo API for the given latitude and longitude.

    Args:
    coordinates (tuple): The latitude of the location.

    Returns:
    float: The current temperature in the coordinates you've asked for
    """

Function:
def get_coordinates_from_city(city_name):
    """
    Fetches the latitude and longitude of a given city name using the Maps.co Geocoding API.

    Args:
    city_name (str): The name of the city.

    Returns:
    tuple: The latitude and longitude of the city.
    """

User Query: {query}<human_end>

'''

EXAMPLE_QUERY = "What's the weather like in Seattle right now?"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.pipeline = pipeline(
            "text-generation",
            model=MODEL_NAME,
            torch_dtype="auto",
            device_map="auto",
        )

    def predict(
        self,
        prompt_template: str = Input(
            description="Prompt template to use for generating the output",
            default=EXAMPLE_PROMPT,
        ),
        query: str = Input(
            description="User query to generate function calls for",
            default=EXAMPLE_QUERY,
        ),
        max_new_tokens: int = Input(
            description="Number of new tokens", ge=1, le=4096, default=2048
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        prompt = prompt_template.format(query=query)
        result = self.pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            return_full_text=False,
            do_sample=False,
            temperature=0.001,
        )[0]["generated_text"]
        return result
