//
// Created by yixuan he on 17/11/2016.
//

#ifndef DL_BASICLAYER_H
#define DL_BASICLAYER_H

#include "../../eigen/Eigen/Dense"
#include <vector>

class Layer{
public:
    virtual std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> input) = 0;
    virtual Eigen::MatrixXd grad(Eigen::MatrixXd* gradient, double lr) = 0;
    virtual void update(std::vector<Eigen::MatrixXd> gradient, double lr) = 0;

protected:
    //TODO: to get gradient, must keep input in memory. Avoic it.
    std::vector<Eigen::MatrixXd> inputs;
};

/**
 * Basic Layer
 * - compute: y = wx+b
 */
class BasicLayer : public Layer{
public:
    BasicLayer(double mean, double variance, int row, int column);
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> input);
    Eigen::MatrixXd grad(Eigen::MatrixXd* gradient, double lr);
    void update(std::vector<Eigen::MatrixXd> gradient, double lr);

    void setActive(const std::string &active);

private:
    Eigen::MatrixXd weight;
    Eigen::MatrixXd bias;
    Eigen::MatrixXd output;

    std::string active = "line";
};

/**
 * L2 Loss Layer
 */
class LossLayer : public Layer {
public:
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> input);
    Eigen::MatrixXd grad(Eigen::MatrixXd* gradient, double lr);
    void update(std::vector<Eigen::MatrixXd> gradient, double lr){}
};


#endif //DL_BASICLAYER_H
