from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    '''This function will return all the requirements in the form of list'''
    requirements=[]
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [req.replace('\n',"") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements=requirements.remove(HYPHEN_E_DOT)
    return requirements        

setup(
name='Student Performance',
version='0.0.1',
author='Venkatesh I',
author_email='iamvenkatesh14@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)