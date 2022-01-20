from setuptools import setup

setup(
    name='radon_transformation',
    version='0.1.1',    
    description='Radon Transformation package',
    author='Hannah Droege',
    license='MIT License',
    packages=['radon_transformation'],
    install_requires=[
                      'numpy',                
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',      
        'Programming Language :: Python :: 3',
    ],
)
