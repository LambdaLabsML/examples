# OpenAI Function Calling with Langchain

## Install

Create a virtual environment and select the kernel `openai-func` from the Jupyter notebook.

```
# Create the virtual environment
git clone https://github.com/LambdaLabsML/examples.git && \
cd examples && \
git checkout langchain && \
cd langchain && \
python -m venv .venv && \
. .venv/bin/activate && \
python -m pip install pip --upgrade --force && \
pip install ipykernel && \
python -m ipykernel install --user --name=openai-func && \
pip install -r requirements.txt


# Launch Jupyter notebook
export OPENAI_API_KEY=<your-openai-api-key> && \
jupyter notebook
```
