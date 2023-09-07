#include <perceptron.h>

int main(){
    double** retornoVal = nullptr;
    double entradasv[3] = {6.8, 9.3, 5.3};
    double salidasEsp[3] = {0.0, 0.0};
    Perceptron miRed(0.1, 3, 2, 2, 2, 2, 0);   //razon de aprendizaje de 0.1, 3entradas, 2salidas, 2capas con (2 y 2) neuronas respectivamente
    miRed.weightCalibrationDefault();
    retornoVal = miRed.runNeuralNetwork(entradasv);
    cout << retornoVal[0][0] << endl;
    cout << retornoVal[0][1] << endl;
    cout << retornoVal[0][2] << endl;
    cout << retornoVal[1][0] << endl;
    cout << retornoVal[1][1] << endl;
    cout << retornoVal[2][0] << endl;
    cout << retornoVal[2][1] << endl;
    miRed.backPropagation(entradasv,salidasEsp);
    miRed.saveWeights("./files/pesosRed.dat");
    miRed.showWeights();
    return 0;
}