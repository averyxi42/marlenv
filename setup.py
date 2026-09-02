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
    ],
    python_requires='>=3.8',

)
