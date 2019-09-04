from setuptools import setup, find_packages

setup(
    name='mtcnn-pytorch',
    version='0.1.0',
    # adding packages
    packages=['mtcnn'],
    package_dir={'mtcnn': 'src/mtcnn'},
    package_data={'mtcnn': ['weights/*.npy']},
    url='',
    license='',
    author='Dan Antoshchenko',
    author_email='antoshchenko.b@gmail.com',
    description=''
)
