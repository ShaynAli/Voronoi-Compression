from setuptools import setup, find_packages

with open('requirements.txt', 'r+') as requirements_file:
    requirements = requirements_file.read().splitlines()

setup(
    name='voronoi-compression',
    version='1.0',
    author='Shayaan Syed Ali',
    author_email='shayaan.syed.ali@gmail.com',
    install_requires=requirements,
    packages=find_packages()
)
