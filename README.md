Use for change voice in video, for fun
# optional, install python3.10 if ness
apt-get update
apt install python3.10 python3.10-venv python3.10-distutils -y
update-alternatives --install /venv/main/bin/python python /usr/bin/python3.10 1
update-alternatives --config python

# install uv
pip install uv