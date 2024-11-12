# setup.py
from setuptools import setup, find_packages

setup(
    name="chai_ml",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tiktoken",
        "pydantic>=2.0.0",
        "openai>=1.0.0",
        "torch",
        "transformers>=4.0.0",
        "ipython",
         "pandas",
        "numpy",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "isort",
            "flake8",
        ],
    },
    author="Ulysses Pascal",
    author_email="upascal@ucla.edu",
    description="A package for text generation using various LLM models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/chai_ml",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)