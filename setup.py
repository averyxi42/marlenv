from setuptools import setup, find_packages
import io

with io.open("README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name='marlenv',
    version='1.0.1',
    url='https://github.com/kc-ml2/marlenv',
    author='Tae Min Ha, Daniel Nam, Won Seok Jung',
    author_email='contact@kc-ml2.com',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        'gymnasium>=1.0',
        'numpy>=1.21',
        'Pillow>=9.0.1',
        # the world models, the search policies, and the data pipeline are
        # part of the project rather than optional extras, so a single
        # install brings up everything the example scripts need
        'torch>=2.0',
        'datasets>=2.14',
        'pygame-ce>=2.4',
    ],
    extras_require={
        'dev': ['pytest>=7.0'],
    },
    python_requires='>=3.9',

)
