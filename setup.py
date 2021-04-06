import setuptools

setuptools.setup(
      name='gym_unbalanced_disk',
      version='0.0.2',
      description='An OpenAI gym environment for unbalanced disk.',
      url="https://github.com/GerbenBeintema/gym-unbalanced-disk",
      author = 'Gerben Beintema',
      author_email = 'g.i.beintema@tue.nl',
      license = 'BSD 3-Clause License',
      python_requires = '>=3.6',
      install_requires = ['gym','numpy','scipy']
      )