from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()


install_requires = [
    ]

scripts = [
    'scripts/debug.py', 
    'scripts/train.py', 
    'scripts/test.py', 
    'scripts/tune.py',
    'demo/app.py',
    'scripts/slurmrun.sh'
]

setup(name='earthnet-models-pytorch', 
        version='0.0.1',
        description="EarthNet Models PyTorch",
        author="Vitus Benson, Claire Robin",
        author_email="{vbenson, crobin}@bgc-jena.mpg.de",
        url="https://earthnet.tech",
        long_description=long_description,
        long_description_content_type="text/markdown",
        classifiers=[
                "Intended Audience :: Science/Research",
                "License :: Other/Proprietary License",
                "Programming Language :: Python :: 3"
                 ],
        packages=find_packages(),
        scripts=scripts,
        install_requires=install_requires,
        )
