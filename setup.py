from setuptools import setup, find_packages

setup(
    name="dinov3-object-detection",
    version="1.0.0",
    description="FCOS-style anchor-free detection heads for frozen DINOv3 backbones",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "torchvision>=0.15",
        "numpy",
        "opencv-python",
        "pycocotools",
        "tqdm",
        "matplotlib",
    ],
)
