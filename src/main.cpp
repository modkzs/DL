#include <iostream>
#include "../eigen/Eigen/Dense"
#include "Layer/BasicLayer.h"
#include "Util/network.h"

int main() {
    std::vector<Layer*> layer;
    LossLayer l;
    BasicLayer b(1, 1, 3, 5);
    b.setActive("sigmod");

    Eigen::MatrixXd x(5, 1);
    x << 1,3,5,2,3;

    Eigen::MatrixXd y(3, 1);
    y << 0,1,0;

//    x.resize(3,1);
//    l.compute(std::vector<Eigen::MatrixXd>{x, y});

    layer.push_back(&b);
    layer.push_back(&l);

    Model m(layer, 0.5);
    for(int i = 0; i < 10; i++) {
        std::cout<< i << " " << m.run(x, y).transpose() << std::endl;
        m.train(x, y);
    }
}