### Setup performed
`uv venv --python 3.9.7`
`./.venv/Scripts/activate`
`git clone https://github.com/sun-umn/PyGRANSO.git`
`cd PyGRANSO`
`uv pip install git+https://github.com/sun-umn/PyGRANSO.git`
`uv pip install -r requirements_cpu.txt`
`cd ..`
`uv pip install -r requirements.txt`
`cd PyGRANSO`
`python test_cpu.py`