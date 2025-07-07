Use for change voice in video, for fun
# install uv
pip install uv

# git 
git clone https://github.com/kurt7courtny/cvv_project.git

# install requirements
cd cvv_project
uv pip install -r requirements.txt

# export env
echo 'export LANGFLOW_COMPONENTS_PATH="/workspace/cvv_project/src"' >> ~/.bashrc
echo 'export PYTHONPATH="$PYTHONPATH:/workspace/cvv_project/src/utils"' >> ~/.bashrc
source ~/.bashrc

# run
langflow run --host 0.0.0.0