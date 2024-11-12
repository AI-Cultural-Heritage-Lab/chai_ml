# chai_ml/generators/openai.py
from openai import OpenAI
from typing import Any, Dict, Optional
from pydantic import BaseModel
import os
from textwrap import dedent
from ..utils.templating import generate_json_template, count_tokens
from .base import BaseTextGenerator

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

    def _max_tokens_adapter(
        self, 
        input_prompt: str, 
        system_prompt: str, 
        max_new_tokens: int
    ) -> int:
        """
        Convert max_new_tokens to max_tokens for OpenAI API.

        Args:
            input_prompt (str): User prompt.
            system_prompt (str): System prompt.
            max_new_tokens (int): Desired number of new tokens.

        Returns:
            int: Total maximum tokens for the API call.
        """
        input_tokens = count_tokens(input_prompt, self.model)
        system_tokens = count_tokens(system_prompt, self.model)
        return max_new_tokens + input_tokens + system_tokens

    def generate_text(
        self,
        input_prompt: str,
        system_prompt: str,
        max_new_tokens: int = 500,
        temperature: float = 0.9,
        top_p: float = 0.9,
        frequency_penalty: float = 0.2,
        presence_penalty: float = 0.0,
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
        max_tokens = self._max_tokens_adapter(
            input_prompt, 
            system_prompt, 
            max_new_tokens
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs
        )

        return response.choices[0].message.content

    def generate_structured_output(
        self,
        input_prompt: str,
        system_prompt: str,
        response_format: BaseModel,
        max_new_tokens: int = 500,
        temperature: float = 0.9,
        top_p: float = 0.9,
        frequency_penalty: float = 0.2,
        presence_penalty: float = 0.0,
        **kwargs
    ) -> Dict:
        """
        Generate structured output using OpenAI's API.

        Args:
            input_prompt (str): User prompt.
            system_prompt (str): System prompt.
            response_format (BaseModel): Pydantic model for response structure.
            max_new_tokens (int): Maximum new tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling parameter.
            frequency_penalty (float): Token frequency penalty.
            presence_penalty (float): Token presence penalty.
            **kwargs: Additional arguments for the API call.

        Returns:
            Dict: Validated structured output.
        """
        template = generate_json_template(response_format)
        
        # Prepare the prompt with JSON structure
        structured_prompt = f"""
        Respond to the following prompt in the provided JSON format.

        # Prompt:
        {input_prompt}

        # JSON Template:
        {json.dumps(template, indent=2)}

        Important: Return only valid JSON data matching the template structure.
        Replace placeholder values with appropriate data.
        """

        max_tokens = self._max_tokens_adapter(
            structured_prompt,
            system_prompt,
            max_new_tokens
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": dedent(structured_prompt)}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            response_format={"type": "json_object"},
            **kwargs
        )

        # Parse and validate the response
        try:
            output = response.choices[0].message.content
            parsed_json = json.loads(output)
            validated_data = response_format.model_validate(parsed_json)
            return validated_data.model_dump(mode='json')
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Error parsing/validating output: {str(e)}")