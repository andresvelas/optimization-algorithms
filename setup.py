from setuptools import setup, find_packages
import os


name = 'default_name'
email = 'default_gmail@example.com'

setup(
    name='base_template_project',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'pytest'
    ],
    author=name,
    author_email=email,
    description='A base template for scientific computing projects',
    python_requires='>=3.7',
)
