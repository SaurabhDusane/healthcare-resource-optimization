from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def _parse_requirements(path):
    """Read a requirements file, ignoring blank lines and (inline) comments."""
    reqs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()  # drop inline/full-line comments
            if line:
                reqs.append(line)
    return reqs


requirements = _parse_requirements("requirements.txt")

setup(
    name="healthcare-resource-optimization",
    version="1.0.0",
    author="Saurabh Dusane",
    author_email="sdusane1@asu.edu",
    description="Healthcare Resource Optimization Analytics Platform with Web Scraping and ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/saurabhdusane/healthcare-resource-optimization",
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
            "health-pipeline=src.pipeline:run_pipeline",
            "health-generate-data=src.data.generate_synthetic_data:main",
        ],
    },
)
