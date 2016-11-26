//
// Created by yixuan he on 20/11/2016.
//

#include "RNNLayer.h"
#include <random>
#include <iostream>

BasicRNNLayer::BasicRNNLayer(double mean, double variance, int row, int column) {
    std::default_random_engine generator;
    std::normal_distribution<double> random(mean, variance);

    w.resize(row, column);
    for(int i = 0; i < row; i ++){
        for(int j = 0; j < column; j++){
            w(i, j) = random(generator);
        }
    }

    b.resize(row, 1);
    for(int i = 0; i < row; i ++){
        b(i, 0) = random(generator);
    }

    u.resize(row, row);
    for(int i = 0; i < row; i ++){
        for(int j = 0; j < row; j++){
            u(i, j) = random(generator);
        }
    }
}

std::vector<Eigen::MatrixXd> BasicRNNLayer::compute(std::vector<Eigen::MatrixXd> target) {
    Eigen::MatrixXd staus;
    std::cout << "BasicRNNLayer" << std::endl;
    for(int i = 0; i < inputs.size(); i++){
        if (i == 0) {
            staus = w * inputs[i] + b;
        } else {
            staus = w * inputs[i] + u*s[i-1] + b;
        }

        s.push_back(staus);
    }

    std::cout << "BasicRNNLayerOver" << std::endl;

    return s;
}

std::vector<Eigen::MatrixXd> BasicRNNLayer::grad(std::vector<Eigen::MatrixXd> gradient, double lr) {
    std::cout << "BasicRNNLayerGrad" << std::endl;
    //gradient to x
    std::vector<Eigen::MatrixXd> g2x;

    //gradient to w
    Eigen::MatrixXd g2w;

    //gradient to b
    Eigen::MatrixXd g2b;

    //gradient to u
    Eigen::MatrixXd g2u;

    for(int i = 0; i < inputs.size(); i++){
        if (i == 0){
            g2w = gradient[i]*inputs[i].transpose();
            g2b = gradient[i];
        } else {
            g2w = gradient[i]*inputs[i].transpose() + u * g2w;
            g2b += gradient[i];
        }

        if (i == 1){
            g2u = gradient[i] * s[i-1].transpose();
        } else if (i > 1){
            g2u = gradient[i] * s[i-1].transpose() + u * g2u;
        }

        g2x.push_back(w.transpose() * gradient[0]);
    }

    update(std::vector<Eigen::MatrixXd>{g2w, g2b, g2u}, lr);

    std::cout << "BasicRNNLayerGradOver" << std::endl;

    return g2x;
}

void BasicRNNLayer::update(std::vector<Eigen::MatrixXd> gradient, double lr) {
    w -= lr * gradient[0];
    b -= lr * gradient[1];
    u -= lr * gradient[2];
}