#include "Matrix.h"
#include "Matrix.cpp"
#include "NeuralNetwork.h"
#include "NeuralNetwork.cpp"
#include <cmath>
#include <cstdlib>

Matrix makeInput(int value)
{
	vector<vector<double>> bits(1, vector<double>(4));
	for (int bit = 0; bit < 4; ++bit)
	{
		bits[0][bit] = static_cast<double>((value >> (3 - bit)) & 1);
	}
	return Matrix(bits);
}

Matrix makeTarget(int value)
{
	vector<vector<double>> target(1, vector<double>(16, 0.0));
	target[0][value] = 1.0;
	return Matrix(target);
}

Matrix runDecoder(const Layer& decoder, const Matrix& input)
{
	return nn_utils::sigmoid(decoder.weights * input.transpose() + decoder.bias)
		.transpose();
}

int argmaxRow(const Matrix& output)
{
	return nn_utils::argmax(output.transpose());
}

int main()
{
	srand(1);
	Layer decoder(4, 16);
	const double weightScale = 1.0 / sqrt(4.0);
	for (int row = 0; row < decoder.weights.getRows(); ++row)
	{
		for (int col = 0; col < decoder.weights.getCol(); ++col)
		{
			const double randomValue = static_cast<double>(rand()) / RAND_MAX;
			decoder.weights.data[row][col] = (randomValue * 2.0 - 1.0) * weightScale;
		}
	}
	decoder.bias = Matrix(16, 1, false);

	const int maxEpochs = 20000;
	const double trainingRate = 1.0;
	int correctCount = 0;
	int epoch = 0;
	for (; epoch < maxEpochs; ++epoch)
	{
		for (int value = 0; value < 16; ++value)
		{
			const Matrix input = makeInput(value);
			const Matrix target = makeTarget(value).transpose();
			const Matrix output = runDecoder(decoder, input).transpose();
			const Matrix delta =
				(output - target).element_multiply(nn_utils::dsigmoid(output));
			const Matrix weightGradient = delta * input;

			decoder.weights = decoder.weights - weightGradient * trainingRate;
			decoder.bias = decoder.bias - delta * trainingRate;
		}

		correctCount = 0;
		for (int value = 0; value < 16; ++value)
		{
			const Matrix output = runDecoder(decoder, makeInput(value));
			if (argmaxRow(output) == value)
			{
				++correctCount;
			}
		}
		if (correctCount == 16)
		{
			break;
		}
	}

	cout << "Training epochs: " << (epoch < maxEpochs ? epoch + 1 : maxEpochs) << endl;
	cout << "Correct patterns: " << correctCount << "/16" << endl;
	for (int value = 0; value < 16; ++value)
	{
		const Matrix output = runDecoder(decoder, makeInput(value));
		cout << value << " -> ";
		output.print_matrix();
		cout << "Predicted output index: " << argmaxRow(output) << endl;
	}

	return correctCount == 16 ? 0 : 1;
}
