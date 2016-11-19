//
// Created by yixuan he on 18/11/2016.
//

#include <iostream>
# include "network.h"

Eigen::MatrixXd Model::run(Eigen::MatrixXd input, Eigen::MatrixXd target) {
    std::vector<Eigen::MatrixXd> cur_input = std::vector<Eigen::MatrixXd>({input});

    for (int i = 0; i < layers.size(); i++){
        cur_input.push_back(target);
        cur_input = layers[i]->compute(cur_input);
    }

    return cur_input[0];
}

void Model::train(Eigen::MatrixXd input, Eigen::MatrixXd target) {
    std::vector<Eigen::MatrixXd> cur_input = std::vector<Eigen::MatrixXd>({input});

    for (int i = 0; i < layers.size(); i++){
        cur_input.push_back(target);
        cur_input = layers[i]->compute(cur_input);
    }

    Eigen::MatrixXd *g = nullptr;
    for (int i = layers.size() - 1; i >= 0; i--){
        Layer *l = layers[i];
        Eigen::MatrixXd tmp = l->grad(g, lr);
        g = &tmp;
    }
}