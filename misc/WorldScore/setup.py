from setuptools import find_namespace_packages, setup

setup(
    name="worldscore",
    version="1.0.0",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="stanford",
    packages=find_namespace_packages(include=["worldscore*"]),
    python_requires=">=3.8",
    install_requires=[
        "av",
        "python-dotenv",
        "omegaconf",
        "submitit",
        "structlog",
        "pyfiglet",
        "numpy",
        "pillow",
        "opencv-python",
        "scipy",
        "fire",
        "hydra-core",
        "requests",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "worldscore=worldscore.cli:main",
            "worldscore-eval=worldscore.run_evaluate:main",
            "worldscore-analysis=worldscore.run_analysis:main",
            "worldscore-gen=worldscore.run_generate:main",
        ],
    },
)
