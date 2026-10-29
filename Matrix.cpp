/*
@Filename: Matrix.cpp
@Author: Ivan Ling (ivan.ling@soton.ac.uk)
@Version: 1.0

@Date: 27/10/2025
@copyright Copyright (c) 2025
*/


#include "Matrix.h"
#include <cstdio>
#include <cstdlib> // For rand() and RAND_MAX
using namespace std;

Matrix::Matrix()	// creates empty 0*0 matrix 
{
	rows = 0;
	col = 0;
	data.clear();


}

Matrix::Matrix(vector<vector<double>> d) // creates a matrix object with given data	
{
	data = d;
	rows = data.size();
	col = data[0].size();

}

Matrix::Matrix(int r, int c, bool isRandom)	// creates a random matrix object of given size
{
	rows = r;
	col = c;
	data.resize(rows, vector<double>(col));
	if (isRandom)
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < col; j++)
			{
				data[i][j] = static_cast<double>(rand()) / RAND_MAX;			
			}
		}
	}

}

int Matrix::getRows() const { return rows; }
int Matrix::getCol() const { return col; }
vector<vector<double>> Matrix::getData() const { return data; }

void Matrix::print_matrix() const
{
	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < col; c++)
		{
			cout<< std::setprecision(2) << data[r][c];
		}
		cout << " " ;
	}
}

Matrix Matrix::operator*(const Matrix& mat) const
{
	if (col != mat.rows)
	{
		throw invalid_argument("matrix multiplication dimensions not compatible");
	}
	vector<vector<double>> result(rows, vector<double>(mat.col));
	for (int i = 0; i < rows; ++i)
		{
		for (int j = 0; j < mat.col; ++j)
		{
			for (int k = 0; k < col; ++k)
			{
				result[i][j] += data[i][k] * mat.data[k][j];
			}
		}
		
	}
	return Matrix(result);
}

Matrix Matrix::operator+(const Matrix& mat) const
{
	if (rows != mat.rows || col != mat.col)
	{
		throw invalid_argument("matrix addition dimentions not compatible");
	}
	vector<vector<double>> result(rows, vector<double>(col));
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			result[i][j] += data[i][j] + mat.data[i][j];	
		}
		
	}
	return Matrix(result);
}

Matrix Matrix::operator-(const Matrix& mat) const
{
	if (rows != mat.rows || col != mat.col)
	{
		throw invalid_argument("matrix subtraction dimentions not compatible");
	}
		
			
	vector<vector<double>> result(rows, vector<double>(col));
		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < col; ++j)
			{
				result[i][j] += data[i][j] - mat.data[i][j];	
			}
			
		}
		return Matrix(result);
}

Matrix Matrix::operator*(double num) const
{
	vector<vector<double>> result(rows, vector<double>(col));
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			result[i][j] += data[i][j] * num;	
		}
	}
	return Matrix(result);
}

Matrix Matrix::transpose() const
{
	vector<vector<double>> result_data(col, vector<double>(rows));
	for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < col; ++j)
        {
            result_data[j][i] = data[i][j];
        }
    }
	return Matrix(result_data);
}

Matrix Matrix::element_multiply(const Matrix& mat) const
{
	if (rows != mat.rows || col != mat.col)
	{
		throw invalid_argument("matrix element multiplication dimentions not compatible");
	}
	vector<vector<double>> result_data(rows, vector<double>(col));
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			result_data[i][j] += data[i][j] * mat.data[i][j];	
		}
		
	}
	return Matrix(result_data);
}