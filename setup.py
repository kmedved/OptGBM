"""Setup OptGBM package."""

from setuptools import find_packages
from setuptools import setup

setup(
    name="optgbm",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "lightgbm>=4.0.0",
        "numpy",
        "optuna>=3.0.0",
        "scikit-learn>=1.0.0",
        "natsort",
        "pandas",
        "scipy",
        "importlib-metadata; python_version < '3.8'",
    ],
    author="Y-oHr-N",
    author_email="y.ohr.n@gmail.com",
    description="Scikit-learn compatible estimator that tunes hyperparameters in LightGBM with Optuna.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Y-oHr-N/OptGBM",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
