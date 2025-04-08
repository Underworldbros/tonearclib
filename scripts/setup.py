from setuptools import setup, find_packages

setup(
    name="tonearclib",
    version="0.1.0",
    license="MIT",
    author="James Boothe",
    author_email="underworldbros@gmail.com",
    description="A semantic music analysis engine for tonal, rhythmic, and emotional profiling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Underworldbros/tonearclib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "librosa",
        "matplotlib",
        "numpy",
        "soundfile",
    ],
    entry_points={
        "console_scripts": [
            "tonearc=tonearclib.__main__:main"
        ]
    },
    include_package_data=True,
    zip_safe=False
)
