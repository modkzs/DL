//
// Created by yixuan he on 17/11/2016.
//

#ifndef DL_BASICLAYER_H
#define DL_BASICLAYER_H

#include "../../eigen/Eigen/Dense"
#include <vector>

class Layer{
    virtual std::vector<Eigen::MatrixXd> compute(Eigen::MatrixXd input[]) = 0;
    virtual std::vector<Eigen::MatrixXd> grad(Eigen::MatrixXd input[], Eigen::MatrixXd* gradient) = 0;
};

/**
 * Basic Layer
 * - compute: y = wx+b
 */
class BasicLayer : public Layer{
public:
    BasicLayer(double mean, double variance, int row, int column);
    std::vector<Eigen::MatrixXd> compute(Eigen::MatrixXd input[]);
    std::vector<Eigen::MatrixXd> grad(Eigen::MatrixXd input[], Eigen::MatrixXd* gradient);
private:
    Eigen::MatrixXd weight;
    Eigen::MatrixXd bias;
    Eigen::MatrixXd output;
};

/**
 * L2 Loss Layer
 */
class LossLayer : public Layer {
public:
    std::vector<Eigen::MatrixXd> compute(Eigen::MatrixXd input[]);
    std::vector<Eigen::MatrixXd> grad(Eigen::MatrixXd input[], Eigen::MatrixXd* gradient);
};


#endif //DL_BASICLAYER_H
