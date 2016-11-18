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

std::vector<Eigen::MatrixXd> BasicLayer::compute(std::vector<Eigen::MatrixXd> input) {
    Eigen::MatrixXd x = input[0];
    output = weight*x + bias;
    output = output.unaryExpr(std::ptr_fun(_active_map[active]));

    inputs.push_back(x);

    return std::vector<Eigen::MatrixXd>({output});
}

Eigen::MatrixXd BasicLayer::grad(Eigen::MatrixXd* gradient, double lr) {
    Eigen::MatrixXd g = gradient->unaryExpr(std::ptr_fun(_active_grad_map[active]));

    // gradient to w
    Eigen::MatrixXd g_w = g * inputs[0].transpose();

    // gradient to d
    Eigen::MatrixXd g_b = g;

    // gradient to x
    Eigen::MatrixXd g_x = weight.transpose() * g;

    update(std::vector<Eigen::MatrixXd>({g_w, g_b}), lr);

//    std::cout << bias.transpose() <<std::endl;

    return g_x;
}

void BasicLayer::update(std::vector<Eigen::MatrixXd> gradient, double lr) {
    weight -= lr * gradient[0];
    bias -= lr * gradient[1];
}

void BasicLayer::setActive(const std::string &active) {
    BasicLayer::active = active;
}

std::vector<Eigen::MatrixXd> LossLayer::compute(std::vector<Eigen::MatrixXd> input) {
    Eigen::MatrixXd x = input[0];
    Eigen::MatrixXd y = input[1];

    inputs.push_back(x);
    inputs.push_back(y);

    Eigen::MatrixXd output = (x-y);
    return std::vector<Eigen::MatrixXd>({ output.array().square().matrix() });
}

Eigen::MatrixXd LossLayer::grad(Eigen::MatrixXd* gradient, double lr) {
    Eigen::MatrixXd x = inputs[0];
    Eigen::MatrixXd y = inputs[1];

    return (x-y);
}