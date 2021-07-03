from setuptools import setup, find_packages

setup(
    name='alphavantage',
    version='0.9.0',
    description='Tools to allow simple retrieval of data from the alphavantage.com data api',
    author='kohlert',
    license='MIT',
    keywords=['stock', 'alphavantage'],
    # url='https://github.com/kohlert/',
    packages=find_packages(),
    install_requires=['pandas', 'framework'],
)
