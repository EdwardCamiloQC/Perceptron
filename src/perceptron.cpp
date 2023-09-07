#include <perceptron.h>

Perceptron::Perceptron(double reason, uint8_t numInputs, uint8_t numOutputs, uint8_t numLayers, ...):learningReason(reason), numberInputs(numInputs), numberOutputs(numOutputs), numberLayers(numLayers){
    va_list layerList;
    va_start(layerList, numLayers);
    int argu;
    neuronsForLayer = new uint8_t[numberLayers];
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

void Perceptron::weightCalibrationDefault(void){
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

void Perceptron::setWeightCalibrationBase(double baseWeight = 0.5){
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        for(uint8_t nNueron = 0; nNueron < neuronsForLayer[nLayer]; nNueron++){
            if(nLayer == 0){
                for(uint8_t nWeight = 0; nWeight <= numberInputs; nWeight++){
                    layer[nLayer][nNueron][nWeight] = baseWeight;
                }
            }else{
                for(uint8_t nWeight = 0; nWeight < neuronsForLayer[nLayer - 1] + 1; nWeight++){
                    layer[nLayer][nNueron][nWeight] = baseWeight;
                }
            }
        }
    }
}

void Perceptron::settingWeights(const char* route){
    double obtainVal;
    ifstream weightFile(route, ios::in | ios::binary);
    if(weightFile.fail()){
        cout << "No se pudo crear el flujo con el archivo...\n";
    }else{
        for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
            for(uint8_t nNueron = 0; nNueron < neuronsForLayer[nLayer]; nNueron++){
                if(nLayer == 0){
                    for(uint8_t nWeight = 0; nWeight <= numberInputs; nWeight++){
                        weightFile.read(reinterpret_cast<char*>(&obtainVal), sizeof(double));
                        layer[nLayer][nNueron][nWeight] = obtainVal;
                    }
                }else{
                    for(uint8_t nWeight = 0; nWeight < neuronsForLayer[nLayer - 1] + 1; nWeight++){
                        weightFile.read(reinterpret_cast<char*>(&obtainVal), sizeof(double));
                        layer[nLayer][nNueron][nWeight] = obtainVal;
                    }
                }
            }
        }
        weightFile.close();
    }
}

double** Perceptron::runNeuralNetwork(const double* valuesIn){
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

void Perceptron::backPropagation(const double* valuesPrub, const double* valuesDesired){
    runNeuralNetwork(valuesPrub);
    double inputValue = 0.0, valAuxi = 0.0, derivate = 0.0;
    uint8_t mayorLayer = 0;
    for(uint8_t k = 0; k < numberLayers; k++){
        if(neuronsForLayer[k] > mayorLayer){
            mayorLayer = neuronsForLayer[k];
        }
    }
    double sigmaAux[mayorLayer];
    double auxDerivatePri[mayorLayer];
    for(uint8_t k = 0; k < mayorLayer; k++){
        sigmaAux[k] = 1.0;
        auxDerivatePri[k] = 1.0;
    }
    uint8_t posibilities = 1;
    uint8_t tope = 0;
    for(int8_t nLayer = numberLayers - 1; nLayer >= 0; nLayer--){
        if(nLayer == 0){
            tope = numberInputs;
        }else{
            tope = neuronsForLayer[nLayer - 1];
        }
        for(uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            derivate = valuesNeuron[nLayer + 1][nNeuron] * (1 - valuesNeuron[nLayer + 1][nNeuron]);
            if(nLayer == numberLayers - 1){
                sigmaAux[nNeuron] = valuesNeuron[nLayer + 1][nNeuron] - valuesDesired[nNeuron];
            }
            for(uint8_t nWeight = 0; nWeight <= tope; nWeight++){
                for(uint8_t nOut = 0; nOut < posibilities; nOut++){
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
                        for(uint8_t b = 0; b < neuronsForLayer[nLayer]; b++){
                            sigmaAux[b] = auxDerivatePri[b];
                        }
                    }
                }
                valAuxi = 0.0;
            }
        }
        posibilities = neuronsForLayer[nLayer];
    }
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        if(nLayer == 0){
            tope = numberInputs;
        }else{
            tope = neuronsForLayer[nLayer - 1];
        }
        for(uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            for(uint8_t nWeight = 0; nWeight <= tope; nWeight++){
                layer[nLayer][nNeuron][nWeight] -= weightsChange[nLayer][nNeuron][nWeight] * learningReason;
            }
        }
    }
}

void Perceptron::saveWeights(const char* path){
    uint8_t tope;
    ofstream flujoPesos;
    flujoPesos.open(path, ios::out | ios::binary);
    if(flujoPesos.fail()){
        cout << "No se pudo crear el flujo" << endl;
    }else{
        for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
            if(nLayer == 0){
                tope = numberInputs;
            }else{
                tope = neuronsForLayer[nLayer - 1];
            }
            for(uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
                for(uint8_t nWeight = 0; nWeight <= tope; nWeight++){
                    flujoPesos.write(reinterpret_cast<char *>(&layer[nLayer][nNeuron][nWeight]), sizeof(double));
                }
            }
        }
        flujoPesos.close();
    }
}

void Perceptron::setReasonLearning(double reason){
    learningReason = reason;
}

void Perceptron::showWeights(){
    for(uint8_t nLayer = 0; nLayer < numberLayers; nLayer++){
        for(uint8_t nNeuron = 0; nNeuron < neuronsForLayer[nLayer]; nNeuron++){
            if(nLayer == 0){
                for(uint8_t nWeight = 0; nWeight <= numberInputs; nWeight++){
                    cout << layer[nLayer][nNeuron][nWeight] << endl;
                }
            }else{
                for(uint8_t nWeight = 0; nWeight < neuronsForLayer[nLayer - 1] + 1; nWeight++){
                    cout << layer[nLayer][nNeuron][nWeight] << endl;
                }
            }
        }
    }
}

Perceptron::~Perceptron(){
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
}