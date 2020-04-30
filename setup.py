import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pls", 
    version="1.0.1",
    author="JerryDong",
    author_email="Juncheng.Dong@duke.edu",
    description="Final Project for Spring2020",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Juncheng-Dong/Partial-Least-Square",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)