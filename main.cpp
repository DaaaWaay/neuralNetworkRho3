#include "NeuralNetwork.cpp"
#include "NeuralNetwork.h"
#include "Matrix.cpp"
#include "Matrix.h"
#include <iostream>
#include <fstream> 
#include <stdio.h>
#include <math.h>

int main()
{
    ANN neuralNetwork({4,16});      //initialises the network with input layer, conversion bit, and output layer
    int maxEppoc = 100;
    int trainingRate = 1;
    int eppoc = 0;
    for (eppoc; eppoc < maxEppoc; eppoc++)
    {
        for (int value = 0; value < 16; value++)
        {
            vector<vector<double>> bits(4, vector<double>(1));
            vector<vector<double>> oneHot(16, vector<double>(1));
            for (int bit = 0; bit < 4; ++bit)
            {
                bits[bit][0] = static_cast<double>((value >> (3 - bit)) & 1);
	        }
            for (int hot = 0; hot < 16; hot++)
            {
                if (value == hot)
                {
                    oneHot[hot][0] = 1;
                }
            }
            Matrix input(bits);
            Matrix output(neuralNetwork.feedforward(input));
            Matrix target(oneHot);

            neuralNetwork.update(target, trainingRate);
            int numCorrect = 0;
            if (value == nn_utils::argmax(output))
            {
                numCorrect++;
            }
            if (numCorrect == 16)
            {
                break;
            }
        }  
    }
    cout << "solved after " << eppoc << "epocs" << endl;
    cout <<"*****************" << endl;
    for (int value = 0; value < 16; value++)
        {
            vector<vector<double>> bits(4, vector<double>(1));
            for (int bit = 0; bit < 4; ++bit)
            {
                bits[bit][0] = static_cast<double>((value >> (3 - bit)) & 1);
	        }
            Matrix input(bits);
            Matrix output(neuralNetwork.feedforward(input));
            cout << value <<" | ";
            output.print_matrix();
            cout << " | " << nn_utils::argmax(output) << endl;
        }
     ofstream MyFile("weights.txt");

}
