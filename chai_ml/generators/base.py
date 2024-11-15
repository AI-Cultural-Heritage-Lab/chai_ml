# chai_ml/generators/base.py
from abc import ABC, abstractmethod
from typing import Any, Optional
from pydantic import BaseModel
import warnings

from abc import ABC, abstractmethod
from typing import Any, Dict, Set
import warnings

class BaseTextGenerator(ABC):

    @abstractmethod
    def generate_text(self, input_prompt: str, system_prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_structured_output(self, input_prompt: str, system_prompt: str, response_format: BaseModel, **kwargs) -> Any:
        pass

class ModelFactory:
    _instances = {}
    
    SUPPORTED_LLMs = {
        "meta-llama/Llama-3.2-1B-Instruct": "LlamaTextGenerator",
        "meta-llama/Llama-3.2-3B-Instruct": "LlamaTextGenerator",
        "meta-llama/Llama-3.2-7B-Instruct": "LlamaTextGenerator",
        "meta-llama/Llama-3.1-8B-Instruct": "LlamaTextGenerator",
        "meta-llama/Llama-3.1-70B-Instruct": "LlamaTextGenerator",
        "gpt-4o": "OpenAITextGenerator",
        "gpt-4-turbo": "OpenAITextGenerator",
        "gpt-4o-mini": "OpenAITextGenerator",
        "gpt-3.5-turbo-0125": "OpenAITextGenerator",
    }

    @classmethod
    def get_model(cls, model_name: str) -> BaseTextGenerator:
        if model_name not in cls._instances:
            if model_name not in cls.SUPPORTED_LLMs:
                warnings.warn(
                    f"Model '{model_name}' is not explicitly supported or tested.",
                    category=UserWarning
                )

            if model_name.startswith("meta-llama"):
                from .llama import LlamaTextGenerator
                cls._instances[model_name] = LlamaTextGenerator(model_id=model_name)
            elif model_name.startswith("gpt-"):
                from .openai import OpenAITextGenerator
                cls._instances[model_name] = OpenAITextGenerator(model=model_name)
            else:
                raise ValueError(f"Model '{model_name}' is not recognized.")

        return cls._instances[model_name]

def generate_text(input_prompt: str, system_prompt: str, model_name: str = "meta-llama/Llama-3.2-1B-Instruct", **kwargs) -> str:
    generator = ModelFactory.get_model(model_name)
    return generator.generate_text(input_prompt, system_prompt, **kwargs)

def generate_structured_output(input_prompt: str, system_prompt: str, response_format: BaseModel, 
                             model_name: str = "meta-llama/Llama-3.2-1B-Instruct", **kwargs) -> Any:
    generator = ModelFactory.get_model(model_name)
    return generator.generate_structured_output(input_prompt, system_prompt, response_format=response_format, **kwargs)