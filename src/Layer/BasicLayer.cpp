//
// Created by yixuan he on 17/11/2016.
//

#include "BasicLayer.h"
#include <random>
#include "action.h"

BasicLayer::BasicLayer(double mean, double variance, int row, int column) {
    std::default_random_engine generator;
    std::normal_distribution<double> random(mean, variance);

    weight.resize(row, column);
    for(int i = 0; i < row; i ++){
        for(int j = 0; j < column; j++){
            weight(i, j) = random(generator);
        }
    }

    bias.resize(row, 1);
    for(int i = 0; i < row; i ++){
        bias(i, 0) = random(generator);
    }
}

std::vector<Eigen::MatrixXd> BasicLayer::compute(Eigen::MatrixXd *input) {
    Eigen::MatrixXd x = input[0];
    output = weight*x + bias;
    output = output.unaryExpr(std::ptr_fun(sigmod));

    return std::vector<Eigen::MatrixXd>({output});
}

std::vector<Eigen::MatrixXd> BasicLayer::grad(Eigen::MatrixXd *input, Eigen::MatrixXd* gradient) {
    std::vector<Eigen::MatrixXd> g(2);
    Eigen::MatrixXd x = input[0];

    // gradient to w
    g[0] = (gradient->unaryExpr(std::ptr_fun(sigmod_grad)) * x.transpose());

    // gradient to d
    g[1] = (*gradient);

    return g;
}

std::vector<Eigen::MatrixXd> LossLayer::compute(Eigen::MatrixXd *input) {
    Eigen::MatrixXd x = input[0];
    Eigen::MatrixXd y = input[1];

    Eigen::MatrixXd output = (x-y);

    return std::vector<Eigen::MatrixXd>({ output.array().square().matrix() });
}

std::vector<Eigen::MatrixXd> LossLayer::grad(Eigen::MatrixXd *input, Eigen::MatrixXd* gradient) {
    Eigen::MatrixXd g;

    Eigen::MatrixXd x = input[0];
    Eigen::MatrixXd y = input[1];

    return std::vector<Eigen::MatrixXd>({ (x-y) });
}