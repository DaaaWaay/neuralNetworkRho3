/*
@Filename: NeuralNetwork.cpp
@Author: Ivan Ling (ivan.ling@soton.ac.uk)
@Version: 1.2

@Date: 27/10/2025
@copyright Copyright (c) 2025
*/


#include "NeuralNetwork.h"
#include <cmath>      

using namespace std;


//A layer refers to the neurons in a single layer
//Each will have their own wieghts and biases

Layer::Layer(int input_size, int output_size)
{
	weights = Matrix(output_size, input_size, true);
	bias = Matrix(output_size, 1, true);
}

Layer::Layer() {}


ANN::ANN(vector<int> sizes)
{
	layerSizes = sizes;
	for (size_t i = 0; i + 1 < sizes.size(); ++i)
    {
        layers.emplace_back(sizes[i], sizes[i + 1]);
    }
}

Matrix ANN::feedforward(Matrix input)
{
	activations.clear();
	activations.push_back(input); //First activation layer is always the input
	Matrix current_activation = input;
	
	for (size_t i = 0; i < layers.size(); ++i)
	{
		Matrix weighted_sum = layers[i].weights * current_activation;	//applies the weigts matrix from each layer
        weighted_sum = weighted_sum + layers[i].bias;					//applies the bias from each layer
        current_activation = nn_utils::sigmoid(weighted_sum);			//aplies non linear sigmoid function
        activations.push_back(current_activation);						
	}
		
	return current_activation;
}

double ANN::update(Matrix target, double training_rate)
{
	Matrix error = activations.back() - target;
	float absError = 0;
	for (int i = 0; i < error.getRows(); i++)
	{
		absError += abs(error.data[i][0]);
	}

	// backwars part

	Matrix delta = error.element_multiply(nn_utils::dsigmoid(activations.back()));

	for (size_t i = layers.size() ; i >= 0; i--)
	{
		Matrix previousDelta;
		Matrix weightGradient = delta * activations[i].transpose();
		if (i > 0)
		{
			previousDelta =(layers[i].weights.transpose() * delta).element_multiply(nn_utils::dsigmoid(activations[i]));
		}
		
        layers[i].weights = layers[i].weights - weightGradient * training_rate;		//twiddling knobs

        layers[i].bias = layers[i].bias - delta * training_rate;

        if (i > 0)
        {
            delta = previousDelta;
        }
	}
	
	return absError;	
}


// Utility functions for ANN
Matrix nn_utils::sigmoid(const Matrix& m)
{
	vector<vector<double>> result_data(m.getRows(), vector<double>(m.getCol()));
	for (int i = 0; i < m.getRows(); ++i)
	{
		for (int j = 0; j < m.getCol(); ++j)
		{
			result_data[i][j] = 1.0 / (1.0 + exp(-m.data[i][j]));
		}
	}
	return Matrix(result_data);
}

Matrix nn_utils::dsigmoid(const Matrix& m)
{
	vector<vector<double>> result_data(m.getRows(), vector<double>(m.getCol()));
	for (int i = 0; i < m.getRows(); ++i)
	{
		for (int j = 0; j < m.getCol(); ++j)
		{
			result_data[i][j] = m.data[i][j] * (1.0 - m.data[i][j]);
		}
	}
	return Matrix(result_data);
}

int nn_utils::argmax(const Matrix& m)
{
	if (m.getCol() != 1)
	{
		cout<< "argmax only supports 1-column matrices!";
	}
	double max_val = -99999;
	int max_idx = 0;
	for (int i = 0; i < m.getRows(); i++)
	{
		if (m.data[i][0] > max_val)
		{
			max_val = m.data[i][0];
			max_idx = i;
		}
	}
	return max_idx;
}
