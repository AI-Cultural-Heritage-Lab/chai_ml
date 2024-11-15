# chai_ml/generators/openai.py
from openai import OpenAI
from typing import Any, Dict, Optional
from pydantic import BaseModel
import os
from textwrap import dedent
from ..utils.templating import generate_json_template, count_tokens
from .base import BaseTextGenerator
from typing import Set

class OpenAITextGenerator(BaseTextGenerator):
    def __init__(
        self, 
        model: str = "gpt-4o",
        api_key: Optional[str] = None
    ):
        """
        Initialize OpenAI configuration and client.

        Args:
            model (str): OpenAI model identifier.
            api_key (str, optional): OpenAI API key. If not provided, reads from environment.
        """
        self.model = model
        
        # Try to get API key from different sources
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            try:
                from google.colab import userdata
                self.api_key = userdata.get("OPENAI_API_KEY")
            except ImportError:
                pass

        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it explicitly or "
                "set it in the OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key)

    def generate_text(
        self,
        input_prompt: str,
        system_prompt: str,
        max_new_tokens: int = 500,
        #temperature: float = 0.9,
        #top_p: float = 0.9,
        #frequency_penalty: float = 0.2,
        #presence_penalty: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text using OpenAI's API.

        Args:
            input_prompt (str): User prompt.
            system_prompt (str): System prompt.
            max_new_tokens (int): Maximum new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            frequency_penalty (float): Token frequency penalty.
            presence_penalty (float): Token presence penalty.
            **kwargs: Additional arguments for the API call.

        Returns:
            str: Generated text.
        """
        # max_tokens = self._max_tokens_adapter(
        #     input_prompt, 
        #     system_prompt, 
        #     max_new_tokens
        # )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=max_new_tokens,
            #temperature=temperature,
            #top_p=top_p,
            #frequency_penalty=frequency_penalty,
            #presence_penalty=presence_penalty,
            **kwargs
        )

        return response.choices[0].message.content

    def generate_structured_output(
        self,
        input_prompt: str,
        system_prompt: str,
        response_format: BaseModel,
        max_new_tokens: int = 500,
        #temperature: float = 0.9,
        #top_p: float = 0.9,
        #frequency_penalty: float = 0.2,
        #presence_penalty: float = 0.0,
        **kwargs
    ) -> Dict:
        """Generate structured output using OpenAI's API."""

        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dedent(input_prompt)}
            ],
            response_format=response_format,
            #temperature=temperature,
            max_tokens=max_new_tokens,
            #top_p=top_p,
            #frequency_penalty=frequency_penalty,
            #presence_penalty=presence_penalty,
            **kwargs
        )

        output = response.choices[0].message.parsed
        return output.model_dump(mode='json')