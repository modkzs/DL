#include <iostream>
#include "../eigen/Eigen/Dense"
#include "Layer/BasicLayer.h"
#include "Layer/RNNLayer.h"
#include "DAG/schedular.h"
#include "ThreadPool/SimpleThreadPool.h"

int main() {
    std::vector<Layer*> layer;

    std::vector<Eigen::MatrixXd> x;
    std::vector<Eigen::MatrixXd> y;

    int size = 20;

    for (int i = 0; i < 2; i++){
        x.push_back(Eigen::MatrixXd::Random(20,1));
    }

    for (int i = 0; i < 2; i++){
        Eigen::MatrixXd tmp(1,1);
        tmp(0,0) = i % 2;
        y.push_back(tmp);
    }

    Schedular s;

    BasicRNNLayer rnn_layer(1,1, 10, 20);
    BasicLayer basic_layer(1,1,1, 10);
    basic_layer.setActive("sigmod");
    LossLayer loss_layer;

    int rnn_id = s.addLayer(&rnn_layer);
    int basic_id = s.addLayer(std::vector<int>({rnn_id}), &basic_layer);
    int loss_id = s.addLayer(std::vector<int>({basic_id}), &loss_layer);

//    s.compute(x, y, loss_id);
    s.run(x, y, std::vector<int>({loss_id}));
//    s.train(x, y);

//    int b_id = s.addLayer(&b);
//    int l_id = s.addLayer(std::vector<int>({b_id}), &l);
//    s.compute(std::vector<Eigen::MatrixXd>({x}), std::vector<Eigen::MatrixXd>({y}), l_id);
//    s.train(std::vector<Eigen::MatrixXd>({x}), std::vector<Eigen::MatrixXd>({y}));

}