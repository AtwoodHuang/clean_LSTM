
#include<iostream>
#include<vector>
#include"LSTM.h"


LstmParam LstmNetwork::My_param;
int main()
{
	srand(0);
	int mem_cell_ct = 100;
	int x_dim = 50;
	int y_dim =1;
	LstmNetwork::My_param.ParamInit(mem_cell_ct, x_dim,y_dim);
	std::vector<std::vector<double>> x1(50);
	std::vector<std::vector<double>> x2(50);
	std::vector<std::vector<double>> x3(50);
	std::vector<std::vector<double>> x4(50);
	Matrix y1(1);
	Matrix y2(1);
	Matrix y3(1);
	Matrix y4(1);
	y1[0].assign(1, -0.5);
	y2[0].assign(1, 0.2);
	y3[0].assign(1, 0.1);
	y4[0].assign(1, -0.5);
	std::vector<Matrix> y_list{ y1,y2,y3,y4 };
	LstmNetwork net;
	for (int i = 0; i < 50; ++i)
	{
		x1[i].assign(1, (double)(rand() % 101) / 101);
		x2[i].assign(1, (double)(rand() % 101) / 101);
		x3[i].assign(1, (double)(rand() % 101) / 101);
		x4[i].assign(1, (double)(rand() % 101) / 101);

	}
	std::vector<Matrix> x_list{ x1,x2,x3,x4 };
	for (int i = 0; i < 500; ++i)
	{
		net.NodeListDestroy();
		std::cout << "iter: " << i;
		for (int i = 0; i < 4; ++i)
		{
			net.x_list_add(x_list[i]);
		}
		std::cout << "  y=: ";
		for (int i = 0; i < 4; ++i)
		{
			std::cout << net.NodeList[i].My_state.y[0][0] << ", ";
		}
		std::cout << std::endl;
		net.y_list_is(y_list, bottom_diff);

		LstmNetwork::My_param.apply_diff(0.1);
	}
	system("pause");
}