# Always prefer setuptools over distutils
# io.open is needed for projects that support Python 2.7
# It ensures open() defaults to text mode with universal newlines,
# and accepts an argument to specify the text encoding
# Python 3 only projects can skip this import
from io import open
from os import path

from setuptools import setup  # find_packages

here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


# get reqs
def requirements():
    list_requirements = []
    with open('requirements.txt') as f:
        for line in f:
            list_requirements.append(line.rstrip())
    return list_requirements


setup(
      name='FedOCS',
      version='1.0.0',  # Required
      description='',  # Optional
      long_description='',  # Optional
      long_description_content_type='text/markdown',  # Optional (see note above)
      url='',  # Optional
      author='',  # Optional
      author_email='',  # Optional
      packages=['FedOCS'],
      # packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
      python_requires='>=3.7',
      install_requires=requirements()  # Optional
)
