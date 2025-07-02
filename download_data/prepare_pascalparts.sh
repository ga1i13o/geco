CURRENT_DIR=$(pwd)
mkdir -p <your_dataset_path>
cd <your_dataset_path>

mkdir PASCAL-VOC
cd PASCAL-VOC
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
tar -xf VOCtrainval_03-May-2010.tar
if [ -d VOCdevkit ]; then
    echo "File extracted successfully"
    rm VOCtrainval_03-May-2010.tar
else
    echo "File extraction failed"
fi

mkdir Parts
cd Parts
wget https://roozbehm.info/pascal-parts/trainval.tar.gz
tar -xf trainval.tar.gz 

if [ -d Annotations_Part ]; then
    echo "File extracted successfully"
    rm trainval.tar.gz
else
    echo "File extraction failed"
fi
cd ${CURRENT_DIR}