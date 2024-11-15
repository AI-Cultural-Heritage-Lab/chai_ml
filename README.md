# chai_ml

A Python package for text generation using various LLM models (OpenAI and Llama) with structured output support. Developed by the Cultural Heritage and AI Lab.

## Installation

### From GitHub
```bash
pip install git+https://github.com/AI-Cultural-Heritage-Lab/chai_ml.git
```

For development installation:
```bash
git clone https://github.com/AI-Cultural-Heritage-Lab/chai_ml.git
cd chai_ml
pip install -e ".[dev]"
```

## Quick Start

To import the library in colab use `pip`: 

```
!pip install git+https://github.com/AI-Cultural-Heritage-Lab/chai_ml.git
```

To use the library you can call common functons using  `from chai_ml`:

```python
from chai_ml import generate_text, generate_structured_output
from pydantic import BaseModel
from typing import List, Optional

# Simple text generation
response = generate_text(
    input_prompt="Write a short story about a robot learning to paint.",
    system_prompt="You are a creative writing assistant.",
    model_name="gpt-4o"  # or "meta-llama/Llama-3.2-1B-Instruct"
)

# Structured output generation
class StoryResponse(BaseModel):
    title: str
    plot_summary: str
    characters: List[str]
    themes: List[str]
    word_count: int
    genre: Optional[str] = None

story_data = generate_structured_output(
    input_prompt="Write a story about a robot learning to paint.",
    system_prompt="You are a creative writing assistant.",
    response_format=StoryResponse,
    model_name="gpt-4o"
)
```

## Features

- Support for both OpenAI and Llama models
- Structured output generation with Pydantic validation
- Token counting utilities
- JSON template generation
- Easy model switching through factory pattern

## Todo
- add model parameter settings/checking
- add gpu/model requirment checking 
- add additional models 
- add cost estimates
- add prompt engineering test kit
    - output evaluations
    - time evaluations
    - batch testing 


## Configuration

### OpenAI Setup

If you are using google colab create a secret called `OPENAI_API_KEY`
Then set your secret value to `your-open-ai-api-key`

Alternatively,
Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or in Python:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

### Llama Setup

1. For gated models ensure you have the necessary permissions and model access.

2. Create a HuggingFace token

3. In colab set add `HF_TOKEN` as a secret.



## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Developed by the Cultural Heritage and AI Lab