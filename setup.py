from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="voxpa",
    version="0.1.0",
    packages=find_packages(),  # auto-detects packages in the folder
    install_requires=required,
    author="Yu-Wen Chen",
    author_email="yc4093@columbia.edu",
    description="This repo includes VoxPA for paper OpenEnded: An Open-Response Speech Corpus for Pronunciation Assessment with Human Annotations and ALM Supervision",
    url="https://github.com/yuwchen/OpenEnded",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
