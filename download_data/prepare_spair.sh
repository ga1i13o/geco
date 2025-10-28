CURRENT_DIR=$(pwd)
mkdir -p $1
cd $1
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar  -xf SPair-71k.tar.gz
if [ -d SPair-71k ]; then
    echo "File extracted successfully"
    rm SPair-71k.tar.gz
else
    echo "File extraction failed"
fi
cp ${CURRENT_DIR}/spair*.csv <your_dataset_path>/SPair-71k
cd ${CURRENT_DIR}
