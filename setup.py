from setuptools import setup, find_packages

setup(
    name="adversarial",
    version="0.1.0",
    description="Adversarial agents",
    author="Siddharth Verma",
    author_email="siddharthverma314@gmail.com",
    url="https://github.com/siddharthverma314/adversarial",
    packages=find_packages(),
    scripts=[
        "scripts/adv",
        "scripts/adv2",
        "scripts/diayn",
        "scripts/sac",
        "scripts/waypoint",
        "scripts/maze",
    ],
)
