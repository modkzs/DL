#include <iostream>
#include "../eigen/Eigen/Dense"
#include "Layer/BasicLayer.h"

int main() {
    LossLayer l;
    Eigen::MatrixXd x(3, 1);
    Eigen::MatrixXd y(3, 1);

    x << 1, 2, 3;
    y << 1.5, 2.3, 2.9;

    Eigen::MatrixXd input[] = {x, y};

    std::cout << l.compute(input)[0] << std::endl;
    Eigen::MatrixXd g = l.grad(input, nullptr)[0];
    std::cout << g << std::endl;

    BasicLayer b(1, 1, 3, 5);
    Eigen::MatrixXd x1(5, 1);
    x1 << 1, 2, 3, 4, 5;
    Eigen::MatrixXd output = b.compute(&x1)[0];
    std::cout << output << std::endl;
    std::vector<Eigen::MatrixXd> tmp = b.grad(&x1, &g);
    std::cout << tmp[0] << std::endl;
    std::cout << tmp[1] << std::endl;
}