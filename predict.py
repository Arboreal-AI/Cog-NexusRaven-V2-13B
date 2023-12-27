# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "NexusRaven-V2-13B-GPTQ"
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
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, use_fast=True, cache_dir=MODEL_CACHE
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, device_map="auto", trust_remote_code=False, revision="main"
        )

        self.model = torch.compile(model)

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
    ) -> str:
        """Run a single prediction on the model"""
        prompt = prompt_template.format(query=query)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        output = self.model.generate(
            inputs=input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )
        return self.tokenizer.decode(output[0])
