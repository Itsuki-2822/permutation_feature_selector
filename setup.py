from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='PermutationFeatureSelector',
    version='0.1.4',
    description='A package for calculating permutation importance and selecting features.',
    author='Itsuki Ito',
    author_email='itoitsuki.28@gmail.com',
    url='https://github.com/Itsuki-2822/PermutationFeatureSelector', 
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'lightgbm'
    ],
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.9, <4',
)
