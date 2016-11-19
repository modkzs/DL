//
// Created by yixuan he on 17/11/2016.
//

#ifndef DL_BASICLAYER_H
#define DL_BASICLAYER_H

#include "../../eigen/Eigen/Dense"
#include <vector>

class Layer{
public:
    virtual Eigen::MatrixXd grad(Eigen::MatrixXd* gradient, double lr) = 0;
    virtual void update(std::vector<Eigen::MatrixXd> gradient, double lr) = 0;
    virtual std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> target) = 0;

    void add_input(std::vector<Eigen::MatrixXd> input, std::vector<int> source){
        inputs.reserve(inputs.size() + input.size());
        inputs.insert(inputs.end(), input.begin(), input.end());

        input_source.reserve(input_source.size() + source.size());
        input_source.insert(input_source.end(), source.begin(), source.end());
    }

protected:
    std::vector<Eigen::MatrixXd> inputs;
    std::vector<int> input_source;
};

/**
 * Basic Layer
 * - compute: y = wx+b
 */
class BasicLayer : public Layer{
public:
    BasicLayer(double mean, double variance, int row, int column);
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> target);
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
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> target);
    Eigen::MatrixXd grad(Eigen::MatrixXd* gradient, double lr);
    void update(std::vector<Eigen::MatrixXd> gradient, double lr){}
};


#endif //DL_BASICLAYER_H
