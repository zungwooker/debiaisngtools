from setuptools import setup, find_packages

setup(
    name="debiasingtools",
    version="0.1",
    description="A set of tools for debiasing research.",
    author="zungwooker",
    author_email="zungwooker@gmail.com",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision'
    ],
    python_requires='>=3.6',
)