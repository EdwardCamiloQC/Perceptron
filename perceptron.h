/*
    The next diagram show an example of a perceptron with two inputs, three outputs and four layers. Where in the 
    first layer has three neurons, two neurons in the second layer and four neurons in the layers three and four.

                    l   l   l   l
                    a   a   a   a
                    y   y   y   y
                    e   e   e   e
                    r   r   r   r
                    _   _   _   _
                    1   2   3   4
                ---------------------
                |   ○       ○   ○   |→→→→   output_1
    input_1 →→→→|       ○           |
                |   ○       ○   ○   |→→→→   output_2
    input_2 →→→→|       ○           |
                |   ○       ○   ○   |→→→→   output_3
                ---------------------

    The syntaxis of the params in the constructor in this case is the next... (2,3,4,3,2,3,3,0);
    In the case where the number of the last layer don't are equal to the outputs, automatically the algorym set
    the number correspont to the outputs.
    The reason learning initially is 0.1
    This perceptron uses a funtion named sigmoide.
*/

#ifndef _PERCEPTRON2_H_
    #define _PERCEPTRON2_H_

    #include <iostream>
    #include <fstream>
    #include <math.h>
    #include <cstdarg>
    #include <cstdint>

    using namespace std;

    class Perceptron2{
    private:
        uint8_t numberInputs, numberOutputs, numberLayers;
        double*** layer;
        double**valuesNeuron;
        double*** weightsChange;
        double learningReason;
        int *neuronsForLayer;
    public:
        Perceptron2(uint8_t numInputs, uint8_t numOutputs, uint8_t numLayers, ...);
        void weightCalibrationDefault(void);
        void settingWeight(const char* route);
        double** runNeuralNetwork(const double* valuesIn);
        void backPropagation(const double* valuesPrub, const double* valuesDesired);
        void saveWeights(const char* path);
        void setReasonLearning(double reason);
        ~Perceptron2();
    };

#endif