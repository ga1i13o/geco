CURRENT_DIR=$(pwd)
mkdir -p <your_dataset_path>
cd <your_dataset_path>
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
if [ -f CUB_200_2011.tgz ]; then
    echo "File downloaded successfully"
else
    echo "File download failed"
fi
tar -xvzf CUB_200_2011.tgz
if [ -d CUB_200_2011 ]; then
    echo "File extracted successfully"
    rm CUB_200_2011.tgz
else
    echo "File extraction failed"
fi

mkdir -p <your_dataset_path_cubannotations>
cd <your_dataset_path_cubannotations>
apt-get install gdown
gdown https://drive.google.com/drive/folders/1DnmpG8Owhv9Rmz_KFozqIKVTJCNSbWc9?usp=sharing
unzip data.zip -C data-v2
cd ${CURRENT_DIR}