#include "perceptron.h"

Perceptron2::Perceptron2(uint8_t numInputs, uint8_t numOutputs, uint8_t numLayers, ...):learningReason(0.1), numberInputs(numInputs), numberOutputs(numOutputs), numberLayers(numLayers){
    va_list layerList;
    va_start(layerList, numLayers);
    int argu;
    neuronsForLayer = new int[numberLayers];
    uint8_t index = 0;
    while((argu = va_arg(layerList, int)) != 0){
        if(index == numberLayers - 1){
            neuronsForLayer[index] = numberOutputs;
        }else{
            neuronsForLayer[index] = argu;
        }
        index++;
    }
    va_end(layerList);
    layer = new double**[numberLayers];
    weightsChange = new double**[numberLayers];
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        layer[nLayer] = new double*[neuronsForLayer[nLayer]];
        weightsChange[nLayer] = new double*[neuronsForLayer[nLayer]];
    }
    for (uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        for (uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            if(nLayer == 0){
                layer[nLayer][nNeuron] = new double[numberInputs + 1];
                weightsChange[nLayer][nNeuron] = new double[numberInputs + 1];
            }else{
                layer[nLayer][nNeuron] = new double[neuronsForLayer[nLayer - 1] + 1];
                weightsChange[nLayer][nNeuron] = new double[neuronsForLayer[nLayer -1] + 1];
            }
        }
    }
    valuesNeuron = new double*[numberLayers + 1];
    for(uint8_t nConnect = 0; nConnect <= numberLayers ; nConnect++){
        if(nConnect == 0){
            valuesNeuron[nConnect] = new double[numberInputs];
        }else{
            valuesNeuron[nConnect] = new double[neuronsForLayer[nConnect - 1]];
        }
    }
}

void Perceptron2::weightCalibrationDefault(void){
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        for(uint8_t nNueron = 0; nNueron < neuronsForLayer[nLayer]; nNueron++){
            if(nLayer == 0){
                for(uint8_t nWeight = 0; nWeight <= numberInputs; nWeight++){
                    layer[nLayer][nNueron][nWeight] = 0.5;
                }
            }else{
                for(uint8_t nWeight = 0; nWeight < neuronsForLayer[nLayer - 1] + 1; nWeight++){
                    layer[nLayer][nNueron][nWeight] = 0.5;
                }
            }
        }
    }
}

void Perceptron2::settingWeight(const char* route){
    ofstream weightFile("pesos");
    if(!weightFile){
        cout << "No se pudo crear el flujo con el archivo...\n";
    }
    unsigned int quantityWeights = (numberInputs + 1) * neuronsForLayer[0];
    for(uint32_t i = 1; i < numberLayers; i++){
        quantityWeights += (neuronsForLayer[i - 1] + 1) * neuronsForLayer[i];
    }
    char weightData[quantityWeights];
    weightFile << quantityWeights;
    weightFile.close();
}

double** Perceptron2::runNeuralNetwork(const double* valuesIn){
    double x = 0.0;
    for(uint8_t nConnector = 0; nConnector < numberInputs; nConnector++){
        valuesNeuron[0][nConnector] = valuesIn[nConnector];
    }
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        for(uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            if(nLayer == 0){
                for(uint8_t nWeight = 0; nWeight < numberInputs; nWeight++){
                    x = x + layer[nLayer][nNeuron][nWeight] * valuesNeuron[0][nWeight];
                }
                x = x + layer[nLayer][nNeuron][numberInputs];
            }else{
                for(uint8_t nWeight = 0; nWeight < neuronsForLayer[nLayer - 1]; nWeight++){
                    x = x + layer[nLayer][nNeuron][nWeight] * valuesNeuron[nLayer][nWeight];
                }
                x = x + layer[nLayer][nNeuron][neuronsForLayer[nLayer - 1]];
            }
            valuesNeuron[nLayer + 1][nNeuron] = 1 / (1 + exp(-x));
            x = 0.0;
        }
    }
    return valuesNeuron;
}

void Perceptron2::backPropagation(const double* valuesPrub, const double* valuesDesired){
    runNeuralNetwork(valuesPrub);
    double aux = 0.0, inputValue, valAuxi = 0.0, derivate;
    unsigned int mayorLayer = 0;
    for(int k = 0; k < numberLayers; k++){
        if(neuronsForLayer[k] > mayorLayer){
            mayorLayer = neuronsForLayer[k];
        }
    }
    cout << hex << mayorLayer << endl;
    double sigmaAux[mayorLayer];
    double auxDerivatePri[mayorLayer];
    for(int k = 0; k < mayorLayer; k++){
        sigmaAux[k] = 1.0;
        auxDerivatePri[k] = 1.0;
    }
    int posibilities = 1;
    int tope;
    for(int nLayer = numberLayers - 1; nLayer >= 0; nLayer--){
        if(nLayer == 0){
            tope = numberInputs;
        }else{
            tope = neuronsForLayer[nLayer - 1];
        }
        cout << tope << endl;
        for(int nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            derivate = valuesNeuron[nLayer + 1][nNeuron] * (1 - valuesNeuron[nLayer + 1][nNeuron]);
            if(nLayer == numberLayers - 1){
                sigmaAux[nNeuron] = valuesNeuron[nLayer + 1][nNeuron] - valuesDesired[nNeuron];
            }
            for(int nWeight = 0; nWeight <= tope; nWeight++){
                for(int nOut = 0; nOut < posibilities; nOut++){
                    if(nLayer == numberLayers - 1){
                        valAuxi = sigmaAux[nNeuron];
                    }else{
                        valAuxi += sigmaAux[nOut] * layer[nLayer + 1][nOut][nNeuron];
                    }
                }
                if(nWeight == tope){
                    inputValue = 1.0;
                }else{
                    inputValue = valuesNeuron[nLayer][nWeight];
                }
                weightsChange[nLayer][nNeuron][nWeight] = valAuxi * derivate * inputValue;
                if(nWeight == tope){
                    auxDerivatePri[nNeuron] = weightsChange[nLayer][nNeuron][nWeight];
                    if(nNeuron == neuronsForLayer[nLayer] - 1){
                        for(int b = 0; b < neuronsForLayer[nLayer]; b++){
                            sigmaAux[b] = auxDerivatePri[b];
                        }
                    }
                }
                valAuxi = 0.0;
            }
        }
        posibilities = neuronsForLayer[nLayer];
    }
    for(int nLayer = 0; nLayer < numberLayers; nLayer++){
        if(nLayer == 0){
            tope = numberInputs;
        }else{
            tope = neuronsForLayer[nLayer - 1];
        }
        for(int nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            for(int nWeight = 0; nWeight <= tope; nWeight++){
                layer[nLayer][nNeuron][nWeight] -= weightsChange[nLayer][nNeuron][nWeight] * learningReason;
            }
        }
    }
}

void Perceptron2::saveWeights(const char* path){
    uint8_t tope;
    ofstream flujoPesos;
    flujoPesos.open("pesosRed", ios::trunc);
    if(!flujoPesos){
        cout << "No se pudo crear el flujo" << endl;
    }
    flujoPesos << "{" << endl;
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        flujoPesos << "capa" << nLayer + 1 << ": ";
        if(nLayer == 0){
            tope = numberInputs;
        }else{
            tope = neuronsForLayer[nLayer - 1];
        }
        for(uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            flujoPesos << "[";
            for(uint8_t nWeight = 0; nWeight <= tope; nWeight++){
                flujoPesos << layer[nLayer][nNeuron][nWeight] << ",";
            }
            flujoPesos << "]" << endl;
        }
        flujoPesos << endl;
    }
    flujoPesos << "}" ;
    flujoPesos.close();
}

void Perceptron2::setReasonLearning(double reason){
    learningReason = reason;
}

Perceptron2::~Perceptron2(){
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        for(uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            delete [] layer[nLayer][nNeuron];
            delete [] weightsChange[nLayer][nNeuron];
        }
    }
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        delete [] layer[nLayer];
        delete [] weightsChange[nLayer];
    }
    delete [] layer;
    delete [] weightsChange;
    delete [] neuronsForLayer;
    for(uint8_t nConnect = 0; nConnect <= numberLayers; nConnect++){
        delete [] valuesNeuron[nConnect];
    }
    delete [] valuesNeuron;
    cout << "salio";
}