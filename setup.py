import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gerrypy-wesg52", # Replace with your own username
    version="0.0.1",
    author="Wes Gurnee",
    author_email="rwg97@cornell.edu",
    description="Python library for political districting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/wesg52/gerrypy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)