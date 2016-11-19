#include <iostream>
#include "../eigen/Eigen/Dense"
#include "Layer/BasicLayer.h"
#include "DAG/network.h"
#include "DAG/schedular.h"

int main() {
    std::vector<Layer*> layer;
    LossLayer l;
    BasicLayer b(1, 1, 3, 5);
    b.setActive("sigmod");

    Eigen::MatrixXd x(5, 1);
    x << 1,3,5,2,3;

    Eigen::MatrixXd y(3, 1);
    y << 0,1,0;

    Schedular s;
    int b_id = s.addLayer(&b);
    int l_id = s.addLayer(std::vector<int>({b_id}), &l);
//    s.compute(std::vector<Eigen::MatrixXd>({x}), std::vector<Eigen::MatrixXd>({y}), l_id);
    s.train(std::vector<Eigen::MatrixXd>({x}), std::vector<Eigen::MatrixXd>({y}));
}