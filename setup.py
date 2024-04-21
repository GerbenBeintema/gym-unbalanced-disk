import setuptools
from setuptools import setup, find_namespace_packages

packages = [a for a in find_namespace_packages(where='.') if a.find('gym_unbalanced_disk')==0]

setuptools.setup(
      name='gym_unbalanced_disk',
      version='0.4.2',
      description='An farama gym environment for unbalanced disk.',
      url="https://github.com/GerbenBeintema/gym-unbalanced-disk",
      author = 'Gerben Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      packages=packages,
      install_requires = ['gymnasium','numpy','scipy', 'pygame', 'pyusb', 'libusb', 'libusb-package'],
      include_package_data=True
)
