
#!/usr/bin/env bash


# MANUAL STEPS BEFORE

# increse gpu ram
# increase swap

set -e
exit # dont ru it

# INSTALL GSTREAMER
# https://qengineering.eu/install-gstreamer-1.18-on-raspberry-pi-4.html

sudo apt-get clean
sudo apt-get autoremove

sudo apt-get install libx264-dev libjpeg-dev -y
sudo apt-get install libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     libgstreamer-plugins-bad1.0-dev \
     gstreamer1.0-plugins-ugly \
     gstreamer1.0-tools \
     gstreamer1.0-gl \
     gstreamer1.0-gtk3 -y

# if you have Qt5 install this plugin
sudo apt-get install gstreamer1.0-qt5 -y
# install if you want to work with audio
sudo apt-get install gstreamer1.0-pulseaudio -y




# INSTALL OPENCV on
# taken from https://qengineering.eu/install-opencv-4.4-on-raspberry-pi-4.html

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install cmake gfortran -y
sudo apt-get install python3-dev python3-numpy -y
sudo apt-get install libjpeg-dev libtiff-dev libgif-dev -y
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev -y
sudo apt-get install libgtk2.0-dev libcanberra-gtk* -y
sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev -y
sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev libv4l-dev -y
sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev -y
sudo apt-get install libjasper-dev liblapack-dev libhdf5-dev -y
sudo apt-get install gcc-arm* protobuf-compiler -y
# from 


sudo apt-get install libgstreamer1.0-dev gstreamer1.0-gtk3 -y
sudo apt-get install libgstreamer-plugins-base1.0-dev gstreamer1.0-gl -y
sudo apt-get install protobuf-compiler -y




cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.4.0.zip
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.4.0.zip
unzip opencv.zip
unzip opencv_contrib.zip
mv opencv-4.4.0 opencv
mv opencv_contrib-4.4.0 opencv_contrib



# get version
python3 --version
# get location
which python3.9
# merge VIRTUALENVWRAPPER_PYTHON=location/version
echo "export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3.9" >> ~/.bashrc
# reload profile
source ~/.bashrc


sudo pip3 install virtualenv
sudo pip3 install virtualenvwrapper



echo "export WORKON_HOME=$HOME/.virtualenvs" >> ~/.bashrc
echo "source /usr/local/bin/virtualenvwrapper.sh" >> ~/.bashrc
source ~/.bashrc
mkvirtualenv cv440


cd ~/opencv/
mkdir build
cd build


cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
        -D ENABLE_NEON=ON \
        -D ENABLE_VFPV3=ON \
        -D WITH_OPENMP=ON \
        -D BUILD_ZLIB=ON \
        -D BUILD_TIFF=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_TBB=ON \
        -D BUILD_TBB=ON \
        -D BUILD_TESTS=OFF \
        -D WITH_EIGEN=OFF \
        -D WITH_GSTREAMER=ON \
        -D WITH_V4L=ON \
        -D WITH_LIBV4L=ON \
        -D WITH_VTK=OFF \
        -D WITH_QT=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_opencv_python3=TRUE \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D BUILD_EXAMPLES=OFF ..

# sudo nano /etc/dphys-swapfile
# set CONF_SWAPSIZE=2048 with the Nano text editor
# sudo /etc/init.d/dphys-swapfile stop
# sudo /etc/init.d/dphys-swapfile start

make -j4

sudo make install
sudo ldconfig
# cleaning (frees 300 KB)
make clean
sudo apt-get update


# sudo nano /etc/dphys-swapfile
# set CONF_SWAPSIZE=100 with the Nano text editor
# sudo /etc/init.d/dphys-swapfile stop
# sudo /etc/init.d/dphys-swapfile start

cd ~/.virtualenvs/cv440/lib/python3.9/site-packages
ln -s /usr/local/lib/python3.9/site-packages/cv2/python-3.9/cv2.cpython-39-arm-linux-gnueabihf.so

