//
// Created by yixuan he on 19/11/2016.
//
#include "schedular.h"
#include <algorithm>
#include <iostream>

int Schedular::addLayer(std::vector<int> source, Layer* dest_layer) {
    int dest = layers.size();
    layers.push_back(dest_layer);
    pending.push_back(0);

    for(int i = 0; i < source.size(); i++){
        addEdge(source[i], dest);
    }

    return dest;
}

int Schedular::addLayer(Layer* layer) {
    int dest = layers.size();

    layers.push_back(layer);
    pending.push_back(0);

    return dest;
}

void Schedular::addEdge(int source, int dest) {
    auto iter = edges.find(source);
    std::vector<int> edge;
    if (iter != edges.end()){
        edge = iter->second;
    }
    edge.push_back(dest);
    edges[source] = edge;
    layers[dest];
    pending[dest] += 1;
}

std::vector<Eigen::MatrixXd> Schedular::compute(std::vector<Eigen::MatrixXd> input, std::vector<Eigen::MatrixXd> target, int output) {
    std::vector<int> ready;
    for(int i = 0; i < pending.size(); i++){
        if (pending[i] == 0){
            ready.push_back(i);

            std::vector<int> source(input.size());
            std::fill(source.begin(), source.end(), -1);
            layers[i]->add_input(input, source);
        }
    }

    while (!ready.empty()) {
        int layer_id = ready[0];
        ready.erase(ready.begin());
        std::vector<Eigen::MatrixXd> out = layers[layer_id]->compute(target);
        std::vector<int> dests = edges[layer_id];

        if (layer_id == output){
            return out;
        }

        for (int i = 0; i < dests.size(); i++){
            int dest = dests[i];
            pending[dest] -= 1;
            if (pending[dest] == 0){
                ready.push_back(dest);
            }

            std::vector<int> source(out.size());
            std::fill(source.begin(), source.end(), layer_id);
            layers[dest]->add_input(out, source);
        }
    }

}

void Schedular::train(std::vector<Eigen::MatrixXd> input, std::vector<Eigen::MatrixXd> y) {}
