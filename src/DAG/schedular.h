//
// Created by yixuan he on 19/11/2016.
//

#ifndef DL_SCHEDULAR_H
#define DL_SCHEDULAR_H

#include <vector>
#include <map>
#include "../Layer/BasicLayer.h"

class Schedular{
private:
    std::vector<Layer*> layers;
    std::vector<int> pending;
    std::map<int, std::vector<int>> edges;

public:
    int addLayer(std::vector<int> input, Layer* layer);
    int addLayer(Layer* layer);
    std::vector<Eigen::MatrixXd> compute(std::vector<Eigen::MatrixXd> input, std::vector<Eigen::MatrixXd> target, int output);
    void train(std::vector<Eigen::MatrixXd> input, std::vector<Eigen::MatrixXd> y);

    // if dest == -1, means is input, don't need to compute it.
    void addEdge(int source, int dest);
};

class Edge{
private:
    int source;
    int destination;

public:
    Edge(int s, int d):source(s), destination(d){}
};

#endif //DL_SCHEDULAR_H
