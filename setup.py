from setuptools import setup

setup(
    name="pablo",
    version="0.0.1",
    author="Tristan Struthers",
    description="An emoji painter",
    url="https://github.com/trstruth/shigo",
    install_requires=[
        "numpy",
        "imageio",
        "pillow",
        "gym",
    ],
    test_suite="nose_collector",
    tests_require=["nose"]
    )
