from setuptools import setup, find_packages

setup(
    name="customer-feedback-nlp-system",
    version="1.0.0",
    description="Production-grade NLP/ML system for customer feedback analysis",
    author="Manuel Corona",
    author_email="manuel.corona@gmail.com",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "pytorch-lightning>=2.1.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.1.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
        ],
        "api": [
            "fastapi>=0.109.0",
            "uvicorn>=0.27.0",
        ],
        "demo": [
            "streamlit>=1.29.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Customer Service/Support",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
