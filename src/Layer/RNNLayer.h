//
// Created by yixuan he on 20/11/2016.
//

#ifndef DL_RNNLAYER_H
#define DL_RNNLAYER_H

#include "BasicLayer.h"

class BasicRNNLayer : public Layer{
public:
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> target);
    std::vector<Eigen::MatrixXd> grad(std::vector<Eigen::MatrixXd> gradient, double lr);
    void update(std::vector<Eigen::MatrixXd> gradient, double lr);

    BasicRNNLayer(double mean, double variance, int row, int column);
private:
    Eigen::MatrixXd w;
    Eigen::MatrixXd b;
    Eigen::MatrixXd u;
    int truncation;

    // store s and used to compute gradient
    std::vector<Eigen::MatrixXd> s;
};

#endif //DL_RNNLAYER_H
