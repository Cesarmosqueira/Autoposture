from setuptools import setup, find_packages

setup(
    name='your_project_name',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here.
        # For example: 'requests', 'numpy', etc.
    ],
    # Add additional fields as needed:
    author='Your Name',
    author_email='your.email@example.com',
    description='A short description of your project',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='http://example.com/your_project',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # If your package has scripts that should be made available to the command line, list them here.
    entry_points={
        'console_scripts': [
            'app=client.start'
        ],
    },
    # If you have data files that need to be included in your packages, specify them here.
    include_package_data=True,
    package_data={
        # Include any *.txt or *.rst files found in the 'your_package' package, for example:
        # 'your_package': ['*.txt', '*.rst'],
    },
    # If your project contains static resources, include them here.
    data_files=[
        # ('my_data', ['data/data_file']),  # Optional: include data files
    ],
)

