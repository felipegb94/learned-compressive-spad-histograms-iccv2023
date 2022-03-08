mex -I/usr/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_core getConstraintsMatrix.cpp mexBase.cpp
mex -I/usr/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_core getContinuousConstraintMatrix.cpp mexBase.cpp 
mex -I/usr/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_core getGridLLEMatrix.cpp mexBase.cpp LLE.cpp 
mex -I/usr/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_core getGridLLEMatrixNormal.cpp mexBase.cpp LLE.cpp
mex -I/usr/include/ANN -L/usr/lib/ -lann -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_core getNormalConstraintMatrix.cpp mexBase.cpp



