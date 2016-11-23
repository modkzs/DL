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

std::vector<Eigen::MatrixXd> BasicLayer::grad(std::vector<Eigen::MatrixXd> gradient, double lr) {
    std::cout << "BasicLayerGrad" << std::endl;
    std::vector<Eigen::MatrixXd> g_x;
    for (int i = 0; i < inputs.size(); i++) {

        Eigen::MatrixXd g = gradient[i].unaryExpr(std::ptr_fun(_active_grad_map[active]));

        // gradient to w
        Eigen::MatrixXd g_w = g * inputs[i].transpose();

        // gradient to d
        Eigen::MatrixXd g_b = g;

        // gradient to x
        g_x.push_back(weight.transpose() * g);

        update(std::vector<Eigen::MatrixXd>({g_w, g_b}), lr);
    }

//    std::cout << bias.transpose() <<std::endl;

    return g_x;
}

std::vector<Eigen::MatrixXd> BasicLayer::compute(std::vector<Eigen::MatrixXd> target) {
    std::cout << "BasicLayer" << std::endl;
    std::vector<Eigen::MatrixXd> output(inputs.size());

    for (int i = 0; i < inputs.size(); i++) {
        Eigen::MatrixXd x = inputs[i];
        output[i] = (weight * x + bias).unaryExpr(std::ptr_fun(_active_map[active]));
    }
    std::cout << "BasicLayer Over" << std::endl;
    return output;

}

void BasicLayer::update(std::vector<Eigen::MatrixXd> gradient, double lr) {
    weight -= lr * gradient[0];
    bias -= lr * gradient[1];
}

void BasicLayer::setActive(const std::string &active) {
    BasicLayer::active = active;
}

std::vector<Eigen::MatrixXd> LossLayer::compute(std::vector<Eigen::MatrixXd> target) {
    std::cout << "LossLayer" << std::endl;
    std::vector<Eigen::MatrixXd> output;
    int size = inputs.size();

    for (int i = 0; i < size; i++) {
        Eigen::MatrixXd x = inputs[i];
        Eigen::MatrixXd y = target[i];

        inputs.push_back(y);

        output.push_back((x - y).array().square().matrix());
    }
    return output;
}

std::vector<Eigen::MatrixXd> LossLayer::grad(std::vector<Eigen::MatrixXd> gradient, double lr) {
    std::cout << "LossLayerGrad" << std::endl;
    int size = inputs.size()/2;

    std::vector<Eigen::MatrixXd> output;

    for(int i = 0; i < size; i++) {
        Eigen::MatrixXd x = inputs[i];
        Eigen::MatrixXd y = inputs[size+i];
        output.push_back(x-y);
    }

    return output;
}
