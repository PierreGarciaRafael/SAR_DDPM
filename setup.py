from setuptools import setup

setup(
    name="guideddiffusion",
    py_modules=["guideddiffusion"],
    version='0.1.0',
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)

