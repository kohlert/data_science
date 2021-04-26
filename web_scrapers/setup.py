from setuptools import setup

setup(
    name='linkedin',
    version='0.1',
    description='A package to assist in collecting data from linkedin pages.',
    author='kohlert',
    license='MIT',
    keywords='webscraper job search linkedin selenium',
    url='https://github.com/kohlert/data_science/tree/master/Web_Scrapers/linkedin',
    packages=['linkedin', 'linkedin.local_drivers'],
    install_requires=['selenium', 'pandas'],
)
