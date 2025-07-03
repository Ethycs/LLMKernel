"""
Setup script for LLM Kernel
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "LLM Jupyter Kernel with Context Management"

# Read version from __init__.py
def get_version():
    init_path = os.path.join('llm_kernel', '__init__.py')
    with open(init_path, 'r') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip('"\'')
    return '0.1.0'

setup(
    name='llm-kernel',
    version=get_version(),
    description='A Jupyter kernel with LiteLLM integration and intelligent context management',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='LLM Kernel Team',
    author_email='',
    url='https://github.com/your-org/llm-kernel',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=[
        'ipykernel>=6.0.0',
        'ipython>=7.0.0',
        'ipywidgets>=7.0.0',
        'litellm>=1.0.0',
        'python-dotenv>=0.19.0',
        'requests>=2.25.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-asyncio>=0.18.0',
            'pytest-mock>=3.6.0',
            'black>=21.0.0',
            'flake8>=3.8.0',
            'mypy>=0.910',
        ],
        'semantic': [
            'sentence-transformers>=2.0.0',
            'scikit-learn>=1.0.0',
            'numpy>=1.20.0',
        ],
        'visualization': [
            'matplotlib>=3.3.0',
            'networkx>=2.5.0',
            'plotly>=5.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'llm-kernel-install=llm_kernel.install:main',
        ],
    },
    include_package_data=True,
    package_data={
        'llm_kernel': ['*.json'],
    },
    data_files=[
        ('share/jupyter/kernels/llm_kernel', ['kernel.json']),
    ],
    zip_safe=False,
    keywords='jupyter kernel llm ai context-management litellm',
)
