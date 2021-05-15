from distutils.core import setup
setup(
  name = 'baayes',         # How you named your package folder (MyLib)
  packages = ['baayes'],   # Chose the same as "name"
  version = '0.1.4',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Bayesian Optimization',   # Give a short description about your library
  author = 'Arpit Jain',                   # Type in your name
  author_email = 'arpit2602@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/arpitjain2602/baayes',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/arpitjain2602/baayes/archive/refs/tags/0.1.4.tar.gz',    # I explain this later on
  keywords = ['Bayesian'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'pandas',
          'numpy','scikit-learn==0.22.1','hyperopt==0.2.3','lightgbm','imbalanced-learn==0.8.0','xgboost==0.90'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
  ],
)