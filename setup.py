from setuptools import find_packages, setup

setup(
    name="d5rl",
    packages=find_packages(
        include=["d5rl"],
    ),
    include_package_data=True,
    version="0.0.1",
    description="Neural NetHack",
    author="Tinkoff AI",
)
