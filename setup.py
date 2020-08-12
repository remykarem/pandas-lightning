import setuptools
from pandas_lightning import __version__

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pandas-lightning",
    version=__version__,
    author="Raimi bin Karim",
    author_email="raimi.bkarim@gmail.com",
    description="Pandas Add-ons",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/remykarem/pandas-lightning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "pandas==1.0.3",
        "scipy==1.4.1",
        "seaborn==0.10.0"
    ]
)
