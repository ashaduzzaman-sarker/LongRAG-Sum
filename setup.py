from setuptools import find_packages, setup

setup(
    name="longragsum",
    version="0.0.1",
    description="Retrieval-Augmented Long-Form Summarization",
    author="Md. Ashaduzzaman Sarker",
    author_email="ashaduzzaman2505@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        line.strip() for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
)