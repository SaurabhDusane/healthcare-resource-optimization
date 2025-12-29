from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="healthcare-resource-optimization",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Healthcare Resource Optimization Analytics Platform with Web Scraping and ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/healthcare-resource-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "black>=23.12.0",
            "pylint>=3.0.3",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "health-scraper=src.scrapers.scheduler:main",
        ],
    },
)
