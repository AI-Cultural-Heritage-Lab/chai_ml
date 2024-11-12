# tests/test_generators.py
import pytest
from chai_ml import LlamaTextGenerator, OpenAITextGenerator, ModelFactory
from pydantic import BaseModel
from typing import List, Optional

class TestResponse(BaseModel):
    title: str
    tags: List[str]
    count: Optional[int] = None

@pytest.fixture
def llama_generator():
    return LlamaTextGenerator("meta-llama/Llama-3.2-1B-Instruct")

@pytest.fixture
def openai_generator():
    return OpenAITextGenerator("gpt-4o")

def test_llama_text_generation(llama_generator):
    response = llama_generator.generate_text(
        input_prompt="Write a short greeting.",
        system_prompt="You are a friendly assistant."
    )
    assert isinstance(response, str)
    assert len(response) > 0

def test_openai_text_generation(openai_generator):
    response = openai_generator.generate_text(
        input_prompt="Write a short greeting.",
        system_prompt="You are a friendly assistant."
    )
    assert isinstance(response, str)
    assert len(response) > 0

def test_model_factory():
    llama = ModelFactory.get_model("meta-llama/Llama-3.2-1B-Instruct")
    openai = ModelFactory.get_model("gpt-4o")
    
    assert isinstance(llama, LlamaTextGenerator)
    assert isinstance(openai, OpenAITextGenerator)

def test_structured_output(openai_generator):
    response = openai_generator.generate_structured_output(
        input_prompt="Write about Python programming.",
        system_prompt="You are a technical writer.",
        response_format=TestResponse
    )
    
    assert isinstance(response, dict)
    assert "title" in response
    assert "tags" in response
    assert isinstance(response["tags"], list)

# tests/test_utils.py
import pytest
from chai_ml.utils.templating import (
    count_tokens,
    generate_json_template,
    validate_template_compatibility,
    merge_template_defaults
)
from pydantic import BaseModel
from typing import List, Optional

class TestModel(BaseModel):
    name: str
    age: int
    tags: List[str]
    description: Optional[str] = None

def test_count_tokens():
    text = "Hello, world!"
    count = count_tokens(text)
    assert isinstance(count, int)
    assert count > 0

def test_generate_json_template():
    template = generate_json_template(TestModel)
    assert isinstance(template, dict)
    assert "name" in template
    assert "age" in template
    assert "tags" in template
    assert "description" in template

def test_template_compatibility():
    template = generate_json_template(TestModel)
    valid_data = {
        "name": "Test",
        "age": 25,
        "tags": ["test"],
        "description": "test"
    }
    assert validate_template_compatibility(template, valid_data)

def test_merge_defaults():
    template = generate_json_template(TestModel)
    partial_data = {
        "name": "Test",
        "age": 25,
        "tags": ["test"]
    }
    merged = merge_template_defaults(template, partial_data)
    assert "description" in merged
    assert merged["description"] is None