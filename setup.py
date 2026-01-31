"""Setup script for video inpainting package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="video-inpainting",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Advanced video inpainting using deep learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/video-inpainting",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
            "pre-commit>=3.3.0",
        ],
        "demo": [
            "streamlit>=1.25.0",
            "gradio>=3.40.0",
        ],
        "advanced": [
            "transformers>=4.30.0",
            "detectron2>=0.6.0",
            "pytorch3d>=0.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "video-inpainting-train=scripts.train:main",
            "video-inpainting-eval=scripts.evaluate:main",
            "video-inpainting-demo=demo.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["configs/*.yaml", "demo/*.py"],
    },
)
