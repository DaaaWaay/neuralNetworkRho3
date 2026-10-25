#include "Matrix.h"
#include "Matrix.cpp"

int main()
{
	Matrix m1(3, 3, true);
	Matrix m2(3, 3, true);

	cout << "m1" << endl;
	m1.print_matrix();
	cout << "m2" << endl;
	m2.print_matrix();

	cout << "m1 * m2" << endl;
	Matrix product = m1 * m2;
	product.print_matrix();
}
