from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="captcha-ocr",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="基于深度学习的验证码识别系统，使用CRNN+CTC模型和FastAPI框架提供服务",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/captcha_ocr",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "pytest-cov>=2.12.1",
            "black>=21.9b0",
            "flake8>=3.9.2",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "captcha-train=scripts.train:main",
            "captcha-predict=scripts.predict:main",
            "captcha-api=scripts.run_api:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)