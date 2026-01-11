rm -rf build dist
python -m build
pip install dist/*.whl --force-reinstall