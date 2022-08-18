from setuptools import find_packages, setup

setup(
    name='ex_setup',
    packages=find_packages(),
    version='0.1.0',
    description='Examples Repo main scripts',
    author='Cooper Lindsey',
    entry_points={
        'console_scripts': [
            'train=train:main',
        ],
    }
)
