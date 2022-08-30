from setuptools import find_packages, setup

from setquence import __version__

setup(
    name="setquence",
    packages=find_packages(),
    version=__version__,
    license="MIT",
    description="SetQuence - Neural Networks for Biological Sequence Set Representations",
    author="Daniel Leon Perinan",
    author_email="daniel.leon-perinan@mailbox.tu-dresden.de",
    url="https://github.com/danilexn/setquence",
    keywords=["artificial intelligence", "attention mechanism", "bioinformatics", "set representation", "sequences",],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
    ],
)
