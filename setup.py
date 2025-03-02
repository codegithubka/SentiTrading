from setuptools import setup, find_packages

setup(
    name="sentitrade",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "praw",
        "pyyaml",
        "scikit-learn",
        "nltk",
        "transformers",
        "sqlalchemy",
        "dash",
        "plotly",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Social Media Sentiment Analysis for Quantitative Trading",
    keywords="trading, sentiment-analysis, nlp, finance",
    url="https://github.com/yourusername/sentitrade",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    python_requires=">=3.8",
)
