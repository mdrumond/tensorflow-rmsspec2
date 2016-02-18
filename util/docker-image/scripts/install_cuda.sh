
mkdir -p /scratch/cuda
cd /scratch/cuda
if [ ! -e "/scratch/cuda/cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb" ]; then
    wget "http://developer.download.nvidia.com/compute/cuda/7_0/Prod/local_installers/rpmdeb/cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb"
fi

sudo dpkg -i cuda-repo-ubuntu1404-7-0-local_7.0-28_amd64.deb
sudo apt-get update && sudo apt-get install -y cuda 

echo "export PATH=/usr/local/cuda-7.0/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc


