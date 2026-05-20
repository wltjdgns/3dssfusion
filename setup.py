from setuptools import setup, find_packages

setup(
    name="kinect_det",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "third_party*"]),
    python_requires=">=3.10",
    install_requires=[
        "pyk4a>=1.4.1",
        "ultralytics>=8.3.40",
        "open3d>=0.18.0",
        "pyyaml>=6.0",
        "numpy>=1.26",
        "opencv-python>=4.10",
    ],
)
