#include "perceptron.h"

void funcion();

int main(){
    funcion();
    //miRed.settingWeight("ruta");
    return 0;
}

void funcion(){
    double** retornoVal = nullptr;
    double entradasv[3] = {6.8, 9.3, 5.3};
    double salidasEsp[3] = {0.0, 0.0};
    Perceptron2 miRed(3,2,2,2,2,0);   //3entradas, 2salidas, 2capas con (2 y 2) neuronas respectivamente
    miRed.weightCalibrationDefault();
    retornoVal = miRed.runNeuralNetwork(entradasv);
    cout << "primero" << endl;
    cout << retornoVal[0][0] << endl;
    cout << retornoVal[0][1] << endl;
    cout << retornoVal[0][2] << endl;
    cout << retornoVal[1][0] << endl;
    cout << retornoVal[1][1] << endl;
    cout << retornoVal[2][0] << endl;
    cout << retornoVal[2][1] << endl;
    miRed.backPropagation(entradasv,salidasEsp);
    retornoVal = miRed.runNeuralNetwork(entradasv);
    cout << "segundo" << endl;
    cout << retornoVal[0][0] << endl;
    cout << retornoVal[0][1] << endl;
    cout << retornoVal[0][2] << endl;
    cout << retornoVal[1][0] << endl;
    cout << retornoVal[1][1] << endl;
    cout << retornoVal[2][0] << endl;
    cout << retornoVal[2][1] << endl;
    //miRed.saveWeights("ruta");
}