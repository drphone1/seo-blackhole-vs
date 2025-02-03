# setup.py
# Created: 2025-02-03 14:23:31 UTC
# Last modified: 2025-02-03 14:23:31 UTC
# Author: drphon
# Repository: drphon/chat-6-deepseek

from setuptools import setup, find_packages
from pathlib import Path
import json

# Read the package version from version.json
with open('version.json') as f:
    VERSION = json.load(f)['version']

# Read the long description from README.md
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Project metadata
PROJECT_URLS = {
    'Source': 'https://github.com/drphon/chat-6-deepseek',
    'Bug Tracker': 'https://github.com/drphon/chat-6-deepseek/issues',
    'Documentation': 'https://github.com/drphon/chat-6-deepseek/wiki'
}

# Package configuration
setup(
    name="web-scraper-advanced",
    version=VERSION,
    author="drphon",
    author_email="drphon@github.com",
    description="An advanced asynchronous web scraping framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/drphon/chat-6-deepseek",
    project_urls=PROJECT_URLS,
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        line.strip()
        for line in Path('requirements.txt').read_text().splitlines()
        if line.strip() and not line.startswith('#')
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.3',
            'pytest-asyncio>=0.21.1',
            'pytest-cov>=4.1.0',
            'black>=23.11.0',
            'mypy>=1.7.0',
            'pylint>=3.0.2',
        ],
        'docs': [
            'sphinx>=7.2.6',
            'sphinx-rtd-theme>=1.3.0',
            'sphinx-autodoc-typehints>=1.24.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'webscraper=web_scraper.main:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

# Last modified: 2025-02-03 14:23:31 UTC
# End of setup.py