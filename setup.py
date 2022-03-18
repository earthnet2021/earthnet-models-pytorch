from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    "tensorboard",
    "numpy",
    "matplotlib",
    "pillow",
    "xarray",
    "zarr",
    "netcdf4",
    "pytorch-lightning",
    "earthnet",
    "segmentation-models-pytorch"
    ]

scripts = [
    'scripts/debug.py', 
    'scripts/train.py', 
    'scripts/test.py', 
    'scripts/tune.py', 
    'scripts/slurmrun.sh'
]

setup(name='earthnet-models-pytorch', 
        version='0.0.1',
        description="EarthNet Models PyTorch",
        author="Vitus Benson",
        author_email="vbenson@bgc-jena.mpg.de",
        url="https://earthnet.tech",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: MIT License",
                "Programming Language :: Python :: 3"
                 ],
        packages=find_packages(),
        scripts=scripts,
        install_requires=install_requires,
        )
