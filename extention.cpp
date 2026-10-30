#include "NeuralNetwork.cpp"
#include "NeuralNetwork.h"
#include "Matrix.cpp"
#include "Matrix.h"
#include <iostream>
#include <fstream> 
#include <stdio.h>
#include <math.h>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

vector<vector<double>> loadCSV(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line;
 
    while (getline(file, line)) {
        replace(line.begin(), line.end(), ',', ' ');  // commas -> spaces
        istringstream ss(line);
 
        vector<double> row;
        double x;
        while (ss >> x) row.push_back(x);
 
        if (!row.empty()) data.push_back(row);  // skip blank lines
    }
    return data;
}


int main()
{
    cout <<"temp"<<endl;
    ANN neuralNetwork({785, 128, 64, 10});      
    int maxEppoc = 100;
    int trainingRate = 1;
    vector<vector<double>> trainingData = loadCSV("fashion-mnist_test.csv");

    for (int i = 0; i < trainingData.size(); i += 5)
    {
        vector<double> inputVecor = trainingData[i];
        vector<vector<double>> inputData(inputVecor.size(), vector<double>(1));

        for (size_t j = 1; j < inputVecor.size(); ++j)
        {
            inputData[j][0] = inputVecor[j];
        }
        vector<vector<double>> tempTarget; 
        for (int hot = 0; hot < 10; hot++)
        {
            if (hot = trainingData[i][0])
            {
                tempTarget[hot][0] = 1;
            }
        }

        Matrix target(tempTarget);
        Matrix input(inputData);
        input = input * (1/256);                                                    //normalise input data to between 0-1 instead of 0-256
        Matrix output(neuralNetwork.feedforward(input));
        neuralNetwork.update(target, 1);
        cout <<"trained line" << i << endl;
    }

    neuralNetwork.saveWeights("weights.txt");
    
    vector<vector<double>> testingData = loadCSV("fashion-mnist_test.csv");

    cout <<"*****************" << endl;
    for (int i = 0; i < testingData.size(); i ++)
    {
        vector<double> inputVecor = testingData[i];
        vector<vector<double>> inputData(inputVecor.size(), vector<double>(1));

        for (size_t j = 1; j < inputVecor.size(); ++j)
        {
            inputData[j][0] = inputVecor[j];
        }
        vector<vector<double>> targetData(1, vector<double>(1));
        targetData[0][0] = testingData[i][0];
        Matrix target(targetData);
        Matrix input(inputData);
        Matrix output(neuralNetwork.feedforward(input));
        
        cout << targetData[0][0] << " ";
        output.print_matrix();
        cout << " " <<endl;
    }
}
