#include"LSTM.h"
#include<cmath>
#include<assert.h>


Matrix bottom_diff(const Matrix &pred, const Matrix &label)
{
	assert(pred.size() == label.size() && pred[0].size() == 1 && label[0].size() == 1);
	Matrix result = pred;
	for (int i = 0; i < pred.size(); ++i)
	{
		result[i][0] = 2.0*(pred[i][0] - label[i][0]);
	}
	return result;
}


Matrix sigmoid(const Matrix& x)
{
	assert(x[0].size() == 1);
	Matrix result = x;
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			result[i][j] = 1.0 / (1.0 + exp(-1.0*x[i][j]));
		}
	}
	return result;
}

Matrix Tanh(const Matrix& x)
{
	assert(x[0].size() == 1);
	Matrix result = x;
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			result[i][j] = tanh(x[i][j]);
		}
	}
	return result;
}

Matrix sigmoid_d(const Matrix& value)
{
	assert(value[0].size() == 1);
	Matrix result = value;
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			result[i][j] = value[i][j] * (1 - value[i][j]);
		}
	}
	return result;
}


Matrix Tanh_d(const Matrix & value)
{
	assert(value[0].size() == 1);
	Matrix result = value;
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			result[i][j] = 1 - value[i][j] * value[i][j];
		}
	}
	return result;
}


Matrix MatrixT(const Matrix& x)
{
	Matrix result(x[0].size());
	for (int i = 0; i < result.size(); ++i)
	{
		result[i].resize(x.size());
	}
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			result[i][j] = x[j][i];
		}
	}
	return result;
}


Matrix MatrixOuter(const Matrix& one, const Matrix& two)
{
	assert(one[0].size() == 1 && two[0].size() == 1);
	Matrix result(one.size());
	for (int i = 0; i < one.size(); ++i)
	{
		result[i].resize(two.size());
	}
	for (int i = 0; i < one.size(); ++i)
	{
		for (int j = 0; j < two.size(); ++j)
		{
			result[i][j] = one[i][0] * two[j][0];
		}
	}
	return result;
}

void NumMutiMatrix(double lr, Matrix &matrix_i, const std::vector<std::vector<double>> &matrix_diff)
{
	assert(matrix_i.size() == matrix_diff.size() && matrix_i[0].size() == matrix_diff[0].size());
	for (int i = 0; i< matrix_i.size(); ++i)
	{
		for (int j = 0; j < matrix_i[0].size(); ++j)
		{
			matrix_i[i][j] -= lr*matrix_diff[i][j];
		}
	}
}

Matrix MatrixMutiMatrix(const Matrix &one, const Matrix &two)
{
	assert(one[0].size() == two.size());
	Matrix result;
	result.resize(one.size());
	for (int i = 0; i < result.size(); ++i)
	{
		result[i].resize(two[0].size());
	}
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			double a = 0.0;
			for (int m = 0; m < two.size(); ++m)
			{
				a += one[i][m] * two[m][j];
			}
			result[i][j] = a;
		}
	}
	return result;
}

Matrix MatrixPluMatrix(const Matrix &one, const Matrix &two)
{
	assert(one.size() == two.size() && one[0].size() == two[0].size());
	Matrix result = one;
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			result[i][j] = one[i][j] + two[i][j];
		}
	}
	return result;
}

Matrix MatrixDianMatrix(const Matrix &one, const Matrix &two)
{
	assert(one.size() == two.size() && one[0].size() == two[0].size());
	Matrix result = one;
	for (int i = 0; i < result.size(); ++i)
	{
		for (int j = 0; j < result[0].size(); ++j)
		{
			result[i][j] = one[i][j] * two[i][j];
		}
	}
	return result;
}

void setMatixToZero(Matrix& matrix)
{
	for (int i = 0; i < matrix.size(); ++i)
	{
		for (int j = 0; j < matrix[0].size(); ++j)
			matrix[i][j] = 0;
	}
}

void LstmParam::apply_diff(double lr)
{
	NumMutiMatrix(lr, wop, wop_diff);
	NumMutiMatrix(lr, wg, wg_diff);
	NumMutiMatrix(lr, wi, wi_diff);
	NumMutiMatrix(lr, wf, wf_diff);
	NumMutiMatrix(lr, wo, wo_diff);

	NumMutiMatrix(lr, bop, bop_diff);
	NumMutiMatrix(lr, bg, bg_diff);
	NumMutiMatrix(lr, bi, bi_diff);
	NumMutiMatrix(lr, bf, bf_diff);
	NumMutiMatrix(lr, bo, bo_diff);

	setMatixToZero(wop_diff);
	setMatixToZero(wg_diff);
	setMatixToZero(wi_diff);
	setMatixToZero(wf_diff);
	setMatixToZero(wo_diff);

	setMatixToZero(bop_diff);
	setMatixToZero(bg_diff);
	setMatixToZero(bi_diff);
	setMatixToZero(bf_diff);
	setMatixToZero(bo_diff);
}

void LstmNode::bottom_data_is(const Matrix &x, const Matrix &s_prev_i, const Matrix &h_prev_i)
{
	h_prev = h_prev_i;
	s_prev = s_prev_i;
	xc = x;
	xc.insert(xc.end(), h_prev.begin(), h_prev.end());

	My_state.g = Tanh(MatrixPluMatrix(MatrixMutiMatrix(LstmNetwork::My_param.wg, xc), LstmNetwork::My_param.bg));
	My_state.i = sigmoid(MatrixPluMatrix(MatrixMutiMatrix(LstmNetwork::My_param.wi, xc), LstmNetwork::My_param.bi));
	My_state.f = sigmoid(MatrixPluMatrix(MatrixMutiMatrix(LstmNetwork::My_param.wf, xc), LstmNetwork::My_param.bf));
	My_state.o = sigmoid(MatrixPluMatrix(MatrixMutiMatrix(LstmNetwork::My_param.wo, xc), LstmNetwork::My_param.bo));
	My_state.s = MatrixPluMatrix(MatrixDianMatrix(My_state.g, My_state.i), MatrixDianMatrix(s_prev, My_state.f));
	My_state.h = MatrixDianMatrix(My_state.s, My_state.o);
	My_state.y = MatrixPluMatrix(MatrixMutiMatrix(LstmNetwork::My_param.wop, My_state.h), LstmNetwork::My_param.bop);

}

void LstmNode::top_diff_is(const Matrix &top_diff_h, const Matrix &top_diff_s, const Matrix &top_diff_y)
{
	Matrix ds = MatrixPluMatrix(MatrixDianMatrix(My_state.o, top_diff_h), top_diff_s);
	Matrix dO = MatrixDianMatrix(My_state.s, top_diff_h);
	Matrix di = MatrixDianMatrix(My_state.g, ds);
	Matrix dg = MatrixDianMatrix(My_state.i, ds);
	Matrix df = MatrixDianMatrix(s_prev, ds);

	Matrix di_input = MatrixDianMatrix(sigmoid_d(My_state.i), di);
	Matrix df_input = MatrixDianMatrix(sigmoid_d(My_state.f), df);
	Matrix do_input = MatrixDianMatrix(sigmoid_d(My_state.o), dO);
	Matrix dg_input = MatrixDianMatrix(Tanh_d(My_state.g), dg);

	LstmNetwork::My_param.wop_diff = MatrixPluMatrix(LstmNetwork::My_param.wop_diff, MatrixOuter(top_diff_y, My_state.h));
	LstmNetwork::My_param.wi_diff = MatrixPluMatrix(LstmNetwork::My_param.wi_diff, MatrixOuter(di_input, xc));
	LstmNetwork::My_param.wf_diff = MatrixPluMatrix(LstmNetwork::My_param.wf_diff, MatrixOuter(df_input, xc));
	LstmNetwork::My_param.wo_diff = MatrixPluMatrix(LstmNetwork::My_param.wo_diff, MatrixOuter(do_input, xc));
	LstmNetwork::My_param.wg_diff = MatrixPluMatrix(LstmNetwork::My_param.wg_diff, MatrixOuter(dg_input, xc));
	LstmNetwork::My_param.bop_diff = MatrixPluMatrix(LstmNetwork::My_param.bop_diff, top_diff_y);
	LstmNetwork::My_param.bi_diff = MatrixPluMatrix(LstmNetwork::My_param.bi_diff, di_input);
	LstmNetwork::My_param.bf_diff = MatrixPluMatrix(LstmNetwork::My_param.bf_diff, df_input);
	LstmNetwork::My_param.bo_diff = MatrixPluMatrix(LstmNetwork::My_param.bo_diff, do_input);
	LstmNetwork::My_param.bg_diff = MatrixPluMatrix(LstmNetwork::My_param.bg_diff, dg_input);

	Matrix dxc = xc;
	for (auto &a : dxc)
	{
		for (auto &b : a)
		{
			b = 0;
		}
	}

	dxc = MatrixPluMatrix(dxc, MatrixMutiMatrix(MatrixT(LstmNetwork::My_param.wi), di_input));
	dxc = MatrixPluMatrix(dxc, MatrixMutiMatrix(MatrixT(LstmNetwork::My_param.wf), df_input));
	dxc = MatrixPluMatrix(dxc, MatrixMutiMatrix(MatrixT(LstmNetwork::My_param.wo), do_input));
	dxc = MatrixPluMatrix(dxc, MatrixMutiMatrix(MatrixT(LstmNetwork::My_param.wg), dg_input));

	My_state.bottom_diff_s = MatrixDianMatrix(ds, My_state.f);
	My_state.bottom_diff_h.assign(dxc.begin() + LstmNetwork::My_param.x_dim, dxc.end());
}

void LstmNetwork::y_list_is(std::vector<Matrix> y, Matrix(*bottom_diff)(const Matrix&, const Matrix&))
{
	int idx = time - 1;
	Matrix diff_y = bottom_diff(NodeList[idx].My_state.y, y[idx]);
	Matrix diff_h = MatrixMutiMatrix(MatrixT(LstmNetwork::My_param.wop), diff_y);
	Matrix diff_s = NodeList[idx].My_state.bottom_diff_s;
	NodeList[idx].top_diff_is(diff_h, diff_s, diff_y);
	idx -= 1;

	while (idx >= 0)
	{
		diff_y = bottom_diff(NodeList[idx].My_state.y, y[idx]);
		diff_h = MatrixMutiMatrix(MatrixT(LstmNetwork::My_param.wop), diff_y);
		diff_h = MatrixPluMatrix(diff_h, NodeList[idx + 1].My_state.bottom_diff_h);
		diff_s = NodeList[idx + 1].My_state.bottom_diff_s;
		NodeList[idx].top_diff_is(diff_h, diff_s, diff_y);
		idx -= 1;
	}
}

void LstmNetwork::x_list_add(const Matrix &x)
{
	LstmState temp(LstmNetwork::My_param.mem_cell_ct, LstmNetwork::My_param.x_dim, LstmNetwork::My_param.y_dim);
	NodeList.push_back(LstmNode(temp));
	time += 1;

	int idx = time - 1;
	if (idx == 0)
	{
		Matrix s_prve(LstmNetwork::My_param.mem_cell_ct);
		Matrix h_prve(LstmNetwork::My_param.mem_cell_ct);
		for (int j = 0; j < LstmNetwork::My_param.mem_cell_ct; ++j)
		{
			s_prve[j].assign(1, 0);
			h_prve[j].assign(1, 0);
		}
		NodeList[idx].bottom_data_is(x, s_prve, h_prve);
	}
	else
	{
		Matrix s_prve = NodeList[idx - 1].My_state.s;
		Matrix h_prve = NodeList[idx - 1].My_state.h;
		NodeList[idx].bottom_data_is(x, s_prve, h_prve);
	}
}