CURRENT_DIR=$(pwd)
mkdir -p <your_dataset_path>
cd <your_dataset_path>
gdown https://drive.google.com/uc?id=1-FNNGcdtAQRehYYkGY1y4wzFNg4iWNad
unzip ap-10k.zip -d ap-10k
rm ap-10k.zip
cd ${CURRENT_DIR}