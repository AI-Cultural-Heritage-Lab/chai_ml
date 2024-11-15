# tests/test_structure_output.py
from chai_ml import generate_structured_output
from pydantic import BaseModel, Field
from typing import List

def test_structured_output():
    class TestResponse(BaseModel):
        title: str = Field(description="Title")
        points: List[str] = Field(description="Points")
        
    response = generate_structured_output(
        input_prompt="Explain Python",
        system_prompt="You are a tutor",
        model_name="gpt-4o",
        response_format=TestResponse
    )
    
    assert isinstance(response, dict)
    assert "title" in response
    assert "points" in response