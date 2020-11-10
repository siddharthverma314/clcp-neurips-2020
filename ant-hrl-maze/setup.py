from setuptools import setup, find_packages

setup(
    name="ant-hrl-maze",
    version="0.1.0",
    description="Ant environments for gym",
    author="Siddharth Verma",
    author_email="siddharthverma314@gmail.com",
    url="https://github.com/siddharthverma314/ant-hrl-maze",
    packages=find_packages(),
    package_data={"ant_hrl_maze": ["assets/*.xml"],},
)
