//
// Created by yixuan he on 18/11/2016.
//

#ifndef DL_NETWORK_H
#define DL_NETWORK_H

#include <vector>
#include "../Layer/BasicLayer.h"

class Model{
public:
    Eigen::MatrixXd run(Eigen::MatrixXd input, Eigen::MatrixXd target);
    void train(Eigen::MatrixXd input, Eigen::MatrixXd target);
    Model(std::vector<Layer*> l, double r): layers(l), lr(r){};
private:
    std::vector<Layer*> layers;
    double  lr;
};

#endif //DL_NETWORK_H
