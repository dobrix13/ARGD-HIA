from setuptools import setup, find_packages

setup(
    name="argd",
    version="0.1.0",
    description="Adaptive Resonant Graph Dynamics for Robust Physiological Signal Processing",
    author="ARGD Research Team",
    packages=find_packages(exclude=["tests", "tests.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "scipy",
        "matplotlib",
        "mne>=1.0.0"
    ],
    python_requires=">=3.10",
)
