# chai_ml/__init__.py
from .generators.llama import LlamaTextGenerator
from .generators.openai import OpenAITextGenerator
from .generators.base import ModelFactory, generate_text, generate_structured_output  # Add these
from .utils.templating import generate_json_template, count_tokens

__version__ = "0.1.0"
__all__ = [
    "LlamaTextGenerator",
    "OpenAITextGenerator",
    "ModelFactory",
    "generate_json_template",
    "count_tokens",
    "generate_text",              # Add these
    "generate_structured_output"   # high-level functions
]