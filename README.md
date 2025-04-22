# One-time setup virtual envirionment:

  ```python -m venv transformer```

  ```source transformer/bin/activate```

  ```pip install -r setup/requirements.txt ```

# One-time setup notebook:

  ```pip install notebook```
  
  ```python -m ipykernel install --user --name=transformer --display-name="Python (transformer)"```

# Regular use virtual environment:

  ```source transformer/bin/activate```
  
# Regular use notebook:

  ```ssh -L 8889:localhost:8889 <your username>@c240g5-110113.wisc.cloudlab.us```
  
  ```jupyter notebook --no-browser --ip=localhost --port=8889```

  > Make sure the "Python (transformer)" kernel is selected in the top-right

# One-time system setup:
  > Only necessary at the start of a new experiment

  ```sh cuda.sh```
