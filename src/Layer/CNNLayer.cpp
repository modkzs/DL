//
// Created by yixuan he on 19/11/2016.
//
#include "CNNLayer.h"
#include <math.h>

std::vector<Eigen::MatrixXd> ConvLayer::compute(std::vector <Eigen::MatrixXd> target) {
    int col = kernel.cols();
    int row = kernel.rows();
    int out_col = inputs[0].cols() - col + 1;
    int out_row = inputs[0].rows() - row + 1;

    std::vector<Eigen::MatrixXd> outputs;

    for(int i = 0; i < inputs.size(); i++) {
        Eigen::MatrixXd x = inputs[i];

        Eigen::MatrixXd output(out_row, out_col);

        for (int i = 0; i < out_row; i++) {
            for (int j = 0; j < out_col; j++) {
                double out = 0;
                for (int m = 0; m < row; m++) {
                    for (int n = 0; n < col; n++) {
                        out += x(i + m, j + n) * kernel(m, n);
                    }
                }
                output(i, j) = out;
            }
        }

        outputs.push_back(output);
    }

    return outputs;
}

std::vector<Eigen::MatrixXd> ConvLayer::grad(std::vector<Eigen::MatrixXd> gradient, double lr) {
    std::vector<Eigen::MatrixXd> grads;

    int g_col = gradient[0].cols();
    int g_row = gradient[0].rows();

    int col = kernel.cols();
    int row = kernel.rows();

    for(int i = 0; i < inputs.size(); i++) {
        Eigen::MatrixXd x = inputs[i];
        Eigen::MatrixXd g = Eigen::ArrayXd::Zero(x.rows(), x.cols());

        for (int i = 0; i < g_row; i++) {
            for (int j = 0; j < g_col; j++) {
                for (int m = 0; m < row; m++) {
                    for (int n = 0; n < col; n++) {
                        g(i + m, j + n) += gradient[0](i, j) * kernel(m, n);
                    }
                }
            }
        }
        grads.push_back(g);
    }

    return grads;
}

std::vector<Eigen::MatrixXd> MaxPool::compute(std::vector<Eigen::MatrixXd> target) {
    int col = inputs[0].cols();
    int row = inputs[0].rows();

    int out_col = ceil(col*1.0/k_col);
    int out_row = ceil(row*1.0/k_row);

    std::vector<Eigen::MatrixXd> outputs;

    for(int i = 0; i < inputs.size(); i++) {
        Eigen::MatrixXd x = inputs[i];
        Eigen::MatrixXd output(out_row, out_col);

        // need to compute pos, so need base
        int base = k_row > k_col ? k_row : k_col;

        for (int i = 0; i < out_row; i++) {
            for (int j = 0; j < out_col; j++) {
                int begin_col = i * k_col;
                int begin_row = j * k_row;

                double max = x(begin_col, begin_row);
                double max_pos = 0;

                int cur_col = begin_col + k_col <= col ? k_col : col - begin_col;
                int cur_row = begin_row + k_row <= row ? k_row : row - begin_row;
                for (int m = 0; m < cur_col; m++) {
                    for (int n = 0; n < cur_row; n++) {
                        if (x(begin_col, begin_row) > max) {
                            max = x(begin_col, begin_row);
                            max_pos = m * base + n;
                        }
                    }
                    output(cur_row, cur_col) = max;
                    kernel(cur_row, cur_col) = max_pos;
                }
            }
        }

        outputs.push_back(output);
    }

    return outputs;
}

std::vector<Eigen::MatrixXd> MaxPool::grad(std::vector<Eigen::MatrixXd> gradient, double lr) {
    int base = k_row > k_col ? k_row : k_col;
    int col = inputs[0].cols();
    int row = inputs[0].rows();

    std::vector<Eigen::MatrixXd> grads(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        Eigen::MatrixXd g(row, col);
        g.setZero();
        Eigen::MatrixXd x = inputs[0];

        for (int i = 0; i < gradient[0].rows(); i++) {
            for (int j = 0; j < gradient[0].cols(); j++) {
                int cur_col = i * k_col + kernel(i, j) / base;
                int cur_row = j * k_row + kernel(i, j) % base;

                g(cur_row, cur_col) = gradient[0](i, j);
            }
        }
        grads[i] = g;
    }

    return grads;
}