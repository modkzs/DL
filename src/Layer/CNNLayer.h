//
// Created by yixuan he on 19/11/2016.
//

#ifndef DL_CNNLAYER_H
#define DL_CNNLAYER_H

#include "BasicLayer.h"

class ConvLayer : public Layer {
public:
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> target);
    std::vector<Eigen::MatrixXd> grad(std::vector<Eigen::MatrixXd> gradient, double lr);
    // no parameter, so no update action
    void update(std::vector<Eigen::MatrixXd> gradient, double lr){};

    ConvLayer(Eigen::MatrixXd k) : kernel(k){};
private:
    Eigen::MatrixXd kernel;
};

class MaxPool : public Layer {
public:
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> target);
    std::vector<Eigen::MatrixXd> grad(std::vector<Eigen::MatrixXd> gradient, double lr);
    void update(std::vector<Eigen::MatrixXd> gradient, double lr){};
    MaxPool(int row, int col) : k_row(row), k_col(col){};

private:
    int k_row;
    int k_col;
    Eigen::MatrixXi kernel;
};

#endif //DL_CNNLAYER_H
