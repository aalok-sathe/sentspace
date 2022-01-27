#!/usr/bin/env python

from setuptools import setup

setup(name='sentspace',
      version='0.0.2',
      description='comaprison of sentences in large sentence spaces along meaningful and latent feature dimensions',
      author='Greta Tuckute <gretatu@mit.edu>, Alvincé Le Arnz Pongos <alvincepongos@gmail.com>, Aalok Sathe <aalok.sathe@richmond.edu> @ EvLab, MIT BCS',
      author_email='gretatu@mit.edu',
      url='https://github.com/aalok-sathe/sentspace.git',
      packages=['sentspace', 'sentspace.web'],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Environment :: Console',
          'Environment :: Web Environment',
          'Framework :: Django',
          'Intended Audience :: Education',
          'Intended Audience :: End Users/Desktop',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Natural Language :: English',
          'Programming Language :: Python',
          'Topic :: Education',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
      ],
      entry_points={
          'console_scripts': [
              'sentspace = sentspace:__main__',
          ],
      },
)
