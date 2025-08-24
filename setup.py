"""
Beverly Knits ERP System - Setup Configuration
Production-ready textile manufacturing ERP with ML forecasting
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="beverly-knits-erp",
    version="2.0.0",
    author="Beverly Knits",
    author_email="erp@beverlyknits.com",
    description="Comprehensive textile manufacturing ERP with ML forecasting and supply chain optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/beverlyknits/erp-system",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Manufacturing",
        "Topic :: Office/Business :: ERP",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.4.0",
        ],
        "ml": [
            "tensorflow>=2.15.0",
            "torch>=2.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "beverly-erp=core.beverly_comprehensive_erp:main",
            "beverly-erp-sync=data_sync.daily_data_sync:main",
            "beverly-erp-validate=scripts.validate_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.html", "*.css", "*.js"],
    },
    project_urls={
        "Bug Reports": "https://github.com/beverlyknits/erp-system/issues",
        "Source": "https://github.com/beverlyknits/erp-system",
        "Documentation": "https://beverlyknits.github.io/erp-docs",
    },
)