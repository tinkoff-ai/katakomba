from setuptools import find_packages, setup

setup(
    name="katakomba",
    packages=find_packages(
        include=["katakomba"],
    ),
    include_package_data=True,
    version="0.0.1",
    description="Neural NetHack",
    author="Tinkoff AI",
)
