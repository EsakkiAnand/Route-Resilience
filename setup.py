from setuptools import setup, find_packages

setup(
    name="route_resilience",
    version="0.1.0",
    description="Satellite Map Healing and Real-Time Road Network Resilience Analysis",
    author="Route Resilience Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "opencv-python",
        "scikit-image",
        "torch",
        "networkx",
        "pyyaml",
        "streamlit",
    ],
)
