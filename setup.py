from setuptools import find_packages, setup

setup(
    name="go2_gym",
    version="1.0.0",
    author="Prakhar Mishra and contributors",
    description=(
        "MorFiC: morphology-conditioned reinforcement learning "
        "for zero-shot quadruped transfer."
    ),
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "ml_logger==0.8.117",
        "ml_dash==0.3.20",
        "jaynes>=0.9.2",
        "params-proto==2.10.5",
        "gym>=0.14.0,<0.26",
        "tqdm",
        "matplotlib",
        "numpy==1.23.5",
    ],
    extras_require={
        "evaluation": [
            "scikit-learn==1.3.2",
            "scipy>=1.7,<1.11",
        ],
        "umap": [
            "umap-learn==0.5.5",
        ],
    },
)