conda create -n geco python=3.10
conda activate geco
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118  # this deletes the torch installation, so we need to reinstall it and cleanup the nvidia installs
pip install jupyterlab
pip install -U matplotlib
pip install transformers
pip install ipympl
pip install triton
pip install open_clip_torch # this deletes the torch installation, so we need to reinstall it and cleanup the nvidia installs
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install hydra-core
pip install -U scikit-learn
pip install pandas
pip install wandb
pip install POT