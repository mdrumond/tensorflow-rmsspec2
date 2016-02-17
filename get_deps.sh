
mkdir -p deps/re2
git clone https://github.com/google/re2.git deps/re2

mkdir -p deps/jpeg-9a
wget -O deps/jpegsrc.v9a.tar.gz "http://www.ijg.org/files/jpegsrc.v9a.tar.gz"
tar  -xvf deps/jpegsrc.v9a.tar.gz -C deps/jpeg-9a/

mkdir -p deps/libpng-1.2.53
wget -O deps/libpng-1.2.53.tar.gz "https://storage.googleapis.com/libpng-public-archive/libpng-1.2.53.tar.gz"
tar -xvf deps/libpng-1.2.53.tar.gz -C deps/libpng-1.2.53/

mkdir -p deps/
wget -O deps/six-1.10.0.tar.gz "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz#md5=34eed507548117b2ab523ab14b2f8b55"
tar -xvf deps/six-1.10.0.tar.gz -C deps/

mkdir -p deps/eigen_archive
wget -O deps/3.3-beta1.tar.gz "https://bitbucket.org/eigen/eigen/get/70505a059011.tar.gz"
tar -xvf deps/3.3-beta1.tar.gz -C deps/eigen_archive

LOCAL_DEPS_DIR=`pwd`/deps

sed "s|__LOCAL_REPO_DIR__|"$LOCAL_DEPS_DIR"|g" WORKSPACE.orig > WORKSPACE
