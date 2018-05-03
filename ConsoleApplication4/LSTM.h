#ifndef LSTM_H
#define LSTM_H
#include<vector>
#include<cstdlib>
#define random (((double)(rand()%101)/50-1)/10)
typedef std::vector<std::vector<double>> Matrix;

Matrix sigmoid(const Matrix& x);
Matrix sigmoid_d(const Matrix& value);
Matrix Tanh(const Matrix& x);
Matrix Tanh_d(const Matrix & value);
Matrix bottom_diff(const Matrix &pred, const Matrix &label);

class LstmParam
{
public:
	void ParamInit(int mem_cell_ct_i, int x_dim_i, int y_dim_i)
	{
		srand(0);
		mem_cell_ct = mem_cell_ct_i;
		x_dim = x_dim_i;
		y_dim = y_dim_i;
		concat_len = x_dim_i + mem_cell_ct_i;
		wop.resize(y_dim);
		wg.resize(mem_cell_ct);
		wi.resize(mem_cell_ct);
		wf.resize(mem_cell_ct);
		wo.resize(mem_cell_ct);

		bop.resize(y_dim);
		bg.resize(mem_cell_ct);
		bi.resize(mem_cell_ct);
		bf.resize(mem_cell_ct);
		bo.resize(mem_cell_ct);

		wop_diff.resize(y_dim);
		wg_diff.resize(mem_cell_ct);
		wi_diff.resize(mem_cell_ct);
		wf_diff.resize(mem_cell_ct);
		wo_diff.resize(mem_cell_ct);

		bop_diff.resize(y_dim);
		bg_diff.resize(mem_cell_ct);
		bi_diff.resize(mem_cell_ct);
		bf_diff.resize(mem_cell_ct);
		bo_diff.resize(mem_cell_ct);

		for (int i = 0; i < y_dim; ++i)
		{
			wop[i].resize(mem_cell_ct);
			wop_diff[i].assign(mem_cell_ct, 0);
			bop[i].resize(1);
			bop[i][0] = random;
			bop_diff[i].assign(1, 0);
			for (int j = 0; j < mem_cell_ct; ++j)
			{
				wop[i][j] = random;
			}
		}

		for (int i = 0; i < mem_cell_ct; ++i)
		{
			wg[i].resize(concat_len);
			wi[i].resize(concat_len);
			wf[i].resize(concat_len);
			wo[i].resize(concat_len);
			wg_diff[i].assign(concat_len, 0);
			wi_diff[i].assign(concat_len, 0);
			wf_diff[i].assign(concat_len, 0);
			wo_diff[i].assign(concat_len, 0);
			bg_diff[i].assign(1, 0);
			bi_diff[i].assign(1, 0);
			bf_diff[i].assign(1, 0);
			bo_diff[i].assign(1, 0);
			bg[i].assign(1, random);
			bi[i].assign(1, random);
			bf[i].assign(1, random);
			bo[i].assign(1, random);
			for (int j = 0; j < concat_len; ++j)
			{
				wg[i][j] = random;
				wi[i][j] = random;
				wf[i][j] = random;
				wo[i][j] = random;
			}
		}

	}

	void apply_diff(double lr);

	int mem_cell_ct;
	int x_dim;
	int concat_len;
	int y_dim;

	Matrix wop;
	Matrix wg;
	Matrix wi;
	Matrix wf;
	Matrix wo;

	Matrix bop;
	Matrix bg;
	Matrix bi;
	Matrix bf;
	Matrix bo;

	Matrix wop_diff;
	Matrix wg_diff;
	Matrix wi_diff;
	Matrix wf_diff;
	Matrix wo_diff;

	Matrix bop_diff;
	Matrix bg_diff;
	Matrix bi_diff;
	Matrix bf_diff;
	Matrix bo_diff;
};


class LstmState
{
public:
	LstmState(int mem_cell_ct, int x_dim, int y_dim)
	{
		y.resize(y_dim);
		g.resize(mem_cell_ct);
		i.resize(mem_cell_ct);
		f.resize(mem_cell_ct);
		o.resize(mem_cell_ct);
		s.resize(mem_cell_ct);
		h.resize(mem_cell_ct);
		bottom_diff_h.resize(mem_cell_ct);
		bottom_diff_s.resize(mem_cell_ct);
		for (int j = 0; j < mem_cell_ct; ++j)
		{
			g[j].assign(1, 0);
			i[j].assign(1, 0);
			f[j].assign(1, 0);
			o[j].assign(1, 0);
			s[j].assign(1, 0);
			h[j].assign(1, 0);
			bottom_diff_s[j].assign(1, 0);
			bottom_diff_h[j].assign(1, 0);
		}
	}

	Matrix y;
	Matrix g;
	Matrix i;
	Matrix f;
	Matrix o;
	Matrix s;
	Matrix h;
	Matrix bottom_diff_h;
	Matrix bottom_diff_s;
};


class LstmNode
{
public:
	LstmNode(const LstmState& a) :My_state(a)
	{
	}

	void bottom_data_is(const Matrix &x, const Matrix &s_prev, const Matrix &h_prev);

	void top_diff_is(const Matrix &top_diff_h, const Matrix &top_diff_s, const Matrix &top_diff_y);


	LstmState My_state;
	Matrix xc;
	Matrix s_prev;
	Matrix h_prev;

};

class LstmNetwork
{
public:
	static LstmParam My_param;
	std::vector<LstmNode> NodeList;
	int time = 0;
	void y_list_is(std::vector<Matrix> y, Matrix(*bottom_diff)(const Matrix&, const Matrix&));
	void x_list_add(const Matrix &x);
	void NodeListDestroy(void)
	{
		time = 0;
		std::vector<LstmNode> temp;
		NodeList = temp;
	}
};
#endif

