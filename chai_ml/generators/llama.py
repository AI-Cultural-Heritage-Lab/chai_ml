# chai_ml/generators/llama.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Any, Dict, Optional
from pydantic import BaseModel, ValidationError
import json
import re
from ..utils.templating import generate_json_template
from .base import BaseTextGenerator

class LlamaTextGenerator(BaseTextGenerator):
    def __init__(self, model_id: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """
        Initialize the Llama model pipeline and tokenizer.

        Args:
            model_id (str): The Hugging Face model ID to load.
        """
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            return_dict=True,
            low_cpu_mem_usage=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Set pad_token_id if not already set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.model.config.eos_token_id

        # Create text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )

    def generate_text(
        self, 
        input_prompt: str, 
        system_prompt: str, 
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Generate text using the Llama model.

        Args:
            input_prompt (str): The user-provided text prompt.
            system_prompt (str): The system-level prompt.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            str: The generated text output.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt},
        ]

        # Format the prompt using chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        response = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            **kwargs
        )

        # Extract the generated text after the assistant marker
        return response[0]["generated_text"].split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[1]

    def generate_structured_output(
        self, 
        input_prompt: str, 
        system_prompt: str, 
        response_format: BaseModel,
        max_new_tokens: int = 400,
        **kwargs
    ) -> Dict:
        """
        Generate structured JSON output validated against a Pydantic model.

        Args:
            input_prompt (str): The user-provided text prompt.
            system_prompt (str): The system-level prompt.
            response_format (BaseModel): Pydantic model defining the response structure.
            max_new_tokens (int, optional): Maximum number of tokens to generate.
            **kwargs: Additional arguments passed to the pipeline.

        Returns:
            Dict: Validated structured output.
        """
        # Generate template from the Pydantic model
        template = generate_json_template(response_format)

        # Construct prompt with JSON format instructions
        structured_prompt = f"""
        Respond to the following prompt in the provided JSON format.

        # Prompt:
        {input_prompt}

        # JSON Template:
        {json.dumps(template, indent=2)}

        Important: Return only valid JSON data matching the template structure.
        Replace placeholder values with appropriate data.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": structured_prompt},
        ]

        # Format prompt and generate response
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate response
        response = self.pipe(
            prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample=True,
            **kwargs
        )

        # Process and validate the response
        raw_response = response[0]["generated_text"].split(
            "<|start_header_id|>assistant<|end_header_id|>"
        )[1]
        
        return self._process_model_output(raw_response, response_format)

    def _process_model_output(
        self, 
        output: str, 
        model: BaseModel
    ) -> Optional[Dict]:
        """
        Process and validate model output against a Pydantic model.

        Args:
            output (str): Raw model output.
            model (BaseModel): Pydantic model for validation.

        Returns:
            Optional[Dict]: Validated data or None if validation fails.
        """
        # Extract JSON using regex
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in model response")

        json_str = json_match.group(0)

        # Handle potential truncation
        if json_str.count('{') > json_str.count('}'):
            json_str += '}' * (json_str.count('{') - json_str.count('}'))

        try:
            # Parse and validate JSON
            parsed_json = json.loads(json_str)
            validated_data = model.model_validate(parsed_json)
            return validated_data.model_dump(mode='json')
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Error parsing/validating output: {str(e)}")