#!/bin/sh -x

#Default means no ode

rootDir=`pwd`
#boost_stage=/usr/local/lib
tinyXML_root=${rootDir}/tinyxmldll
JGTL_include=${rootDir}/JGTL/include


echo Checking if dev-intel07 or dev-intel09 is host name
if [ `hostname` = "dev-intel07" -o `hostname` = "dev-intel09" ]
then
    module unload intelcc/11.0.083
    module load gnu

    CC=gcc 
    CXX=g++

    echo Setting boost variables

#    echo Using Intel compiler
#    module unload hpc-defaults; module load sgicluster
#    CXX=/opt/intel/cce/10.0.025/bin/icpc

elif [[ `hostname` = mmmlog* ]]
then
    echo On Mt. Moran
    module load CMAKE/cmake-2.8.9
#    module load BOOST/boost_1_51_0_gcc
    module load BOOST/boost_1_51_0_intel


    module load INTEL/2012.0.032/compilers

    CC=icc
    CXX=icpc

    export CC=icc
    export CXX=icpc

    echo Setting boost path variables
#    BOOST_INCLUDEDIR=/apps/BOOST/boost_1_51_0_gcc/include
#    BOOST_LIBRARYDIR=/apps/BOOST/boost_1_51_0_gcc/lib
#    boost_stage=/apps/BOOST/boost_1_51_0_gcc/lib
#    Boost_LIBRARIES="/apps/BOOST/boost_1_51_0_gcc/lib/libboost_filesystem.so.1.51.0;/apps/BOOST/boost_1_51_0_gcc/lib/libboost_system.so.1.51.0;/apps/BOOST/boost_1_51_0_gcc/lib/libboost_thread.so.1.51.0"

    BOOST_INCLUDEDIR=/apps/BOOST/boost_1_51_0_intel-linux/include
    BOOST_LIBRARYDIR=/apps/BOOST/boost_1_51_0_intel-linux/lib
    boost_stage=/apps/BOOST/boost_1_51_0_intel-linux/lib/
    Boost_LIBRARIES="/apps/BOOST/boost_1_51_0_intel-linux/lib/libboost_filesystem.so.1.51.0;/apps/BOOST/boost_1_51_0_intel-linux/lib/libboost_system.so.1.51.0;/apps/BOOST/boost_1_51_0_intel-linux/lib/libboost_thread.so.1.51.0"


elif [ `hostname` = "nac93" ]
then
    echo On Nick Desktop
#     module load CMAKE/cmake-2.8.9
# #    module load BOOST/boost_1_51_0_gcc
#     module load BOOST/boost_1_51_0_intel


#     module load INTEL/2012.0.032/compilers

#     CC=icc
#     CXX=icpc

#     export CC=icc
#     export CXX=icpc
    CC=gcc
    CXX=g++

    echo Setting boost path variables
#    BOOST_INCLUDEDIR=/apps/BOOST/boost_1_51_0_gcc/include
#    BOOST_LIBRARYDIR=/apps/BOOST/boost_1_51_0_gcc/lib
#    boost_stage=/apps/BOOST/boost_1_51_0_gcc/lib
#    Boost_LIBRARIES="/apps/BOOST/boost_1_51_0_gcc/lib/libboost_filesystem.so.1.51.0;/apps/BOOST/boost_1_51_0_gcc/lib/libboost_system.so.1.51.0;/apps/BOOST/boost_1_51_0_gcc/lib/libboost_thread.so.1.51.0"

    BOOST_INCLUDEDIR=/usr/include
    BOOST_LIBRARYDIR=/usr/lib
    boost_stage=/usr/lib
    Boost_LIBRARIES="/usr/lib/libboost_filesystem.so;/usr/lib/libboost_system.so;/usr/lib/libboost_thread.so"


elif [ `hostname` = "n178.alice.cme.msu.edu" ]
then

    CC=gcc
    CXX=g++

    BOOST_INCLUDEDIR=/usr/include
    BOOST_LIBRARYDIR=/usr/lib

    echo Setting boost_root
    boost_stage=/usr/lib

elif [ `hostname` = "endlessforms" -o `hostname` = "devendlessforms" ]
then


    CC=gcc
    CXX=g++

    #boost_root=/usr/include/boost
    boost_root=/usr/include

    BOOST_INCLUDEDIR=/usr/include/
    BOOST_LIBRARYDIR=/usr/lib

    echo Setting boost_root
    boost_stage=/usr/lib


elif [ `hostname` = "rodan.cse.msu.edu" ]
then
    echo Configuring for Rodan...which probably will not work
else
    echo Not using valid host.  Using local boost_root
fi


echo Building required packages
echo

# build TinyXML
if test -e ${tinyXML_root}/out/libtinyxmldll.so
then
    echo TinyXML exists.  Skipping
    echo
else
    echo Building TinyXML
    cd ${rootDir}
    cd zlib
    cmake .
    make

    cd ..
    cd tinyxmldll
    cmake .
    make
    cd ${rootDir}

    echo Finished making TinyXML
    echo
fi

echo Not building ode. 

# build HyperNEAT
if test -e ${rootDir}/HyperNEAT/out/Hypercube_NEAT
then
    echo Hypercube_NEAT exists.  Skipping
    echo
else
    echo Building HyperNEAT
    cd ${rootDir}
    #rm -rf CMakeFiles CMakeCache.txt cmake_install.cmake
    rm -rf hyperneatLibs
    mkdir hyperneatLibs
    cd hyperneatLibs 
    #ln -s ${boost_stage}/libboost_thread.so.1.51.0
    #ln -s ${boost_stage}/libboost_filesystem.so.1.51.0
    #ln -s ${boost_stage}/libboost_system.so.1.51.0
    cd ${rootDir}
    cd HyperNEAT
    echo BOOST_LIBRARYDIR is ${BOOST_LIBRARYDIR}
    cmake -DBoost_LIBRARY_DIRS:string=${BOOST_LIBRARYDIR} -DBoost_INCLUDE_DIRS:string=${BOOST_INCLUDEDIR} -DBoost_LIBRARIES=${Boost_LIBRARIES} .
    cmake -DCOMPILE_ODE=OFF -DCMAKE_INSTALL_PREFIX=${rootDir}/out -DJGTL_INCLUDE=${JGTL_include} -DBoost_LIBRARY_DIRS:string=${BOOST_LIBRARYDIR} -DBoost_INCLUDE_DIRS:string=${BOOST_INCLUDEDIR} -DBOOST_ROOT=${boost_root} -DBOOST_STAGE:string=${boost_stage} -DBoost_LIBRARIES=${Boost_LIBRARIES} -DEXECUTABLE_OUTPUT_PATH=${rootDir}/out -DJGTL_INCLUDE=${JGTL_include} -DLIBRARY_OUTPUT_PATH:string=${rootDir}/hyperneatLibs -DTINYXMLDLL_INCLUDE=${tinyXML_root}/include -DTINYXMLDLL_LIB=${tinyXML_root}/out -DENDLESSFORMS=ON

    #make $@ #--debug=v
    make VERBOSE=1 $@ #--debug=v
    cd ${rootDir}

    echo Finished making Hypercube_NEAT
    echo
fi
