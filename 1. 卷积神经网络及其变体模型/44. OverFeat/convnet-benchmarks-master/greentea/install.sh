git clone https://github.com/naibaf7/caffe.git
cd caffe
git checkout master

# Dependencies
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev 
sudo apt-get install -y protobuf-compiler gfortran libjpeg62 libfreeimage-dev libatlas-base-dev git python-dev python-pip 
sudo apt-get install -y libgoogle-glog-dev libbz2-dev libxml2-dev libxslt-dev libffi-dev libssl-dev libgflags-dev liblmdb-dev python-yaml
sudo apt-get install -y libviennacl-dev opencl-headers libopenblas-base libopenblas-dev
easy_install pillow #conda python

# Compile Caffe
cp ../Makefile.config Makefile.config

cores=`grep -c ^processor /proc/cpuinfo`

make all -j$cores VIENNACL_DIR=../ViennaCL-1.7.0/
make test -j$cores VIENNACL_DIR=../ViennaCL-1.7.0/
make runtest -j$cores VIENNACL_DIR=../ViennaCL-1.7.0/
