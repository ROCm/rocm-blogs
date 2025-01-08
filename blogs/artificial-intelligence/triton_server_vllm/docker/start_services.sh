#!/bin/bash
jupyter-lab --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token='' &
tritonserver --model-store /usr/src/app/src/model_repository