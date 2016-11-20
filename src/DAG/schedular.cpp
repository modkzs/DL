//
// Created by yixuan he on 19/11/2016.
//
#include "schedular.h"
#include <algorithm>
#include <iostream>

int Schedular::addLayer(std::vector<int> source, Layer* dest_layer) {
    int dest = layers.size();
    layers.push_back(dest_layer);
    in_edges[dest] = std::vector<int>();
    out_edges[dest] = std::vector<int>();

    for(int i = 0; i < source.size(); i++){
        addEdge(source[i], dest);
    }

    return dest;
}

int Schedular::addLayer(Layer* layer) {
    int dest = layers.size();

    layers.push_back(layer);

    return dest;
}

void Schedular::addEdge(int source, int dest) {
    in_edges[dest].push_back(source);
    out_edges[source].push_back(dest);
}

std::vector<Eigen::MatrixXd> Schedular::compute(std::vector<Eigen::MatrixXd> input, std::vector<Eigen::MatrixXd> target, int output) {
    std::vector<int> ready;
    std::vector<int> pending;

    for(int i = 0; i < in_edges.size(); i++){
        if (in_edges[i].size() == 0){
            ready.push_back(i);

            std::vector<int> source(input.size());
            std::fill(source.begin(), source.end(), -1);
            layers[i]->add_input(input, source);
        }
        pending.push_back(in_edges[i].size());
    }

    while (!ready.empty()) {
        int layer_id = ready[0];
        ready.erase(ready.begin());
        std::vector<Eigen::MatrixXd> out = layers[layer_id]->compute(target);
        std::vector<int> dests = out_edges[layer_id];

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

void Schedular::train(std::vector<Eigen::MatrixXd> input, std::vector<Eigen::MatrixXd> target) {
    std::vector<int> ready;
    std::vector<int> pending;

    for(int i = 0; i < in_edges.size(); i++){
        if (in_edges[i].size() == 0){
            ready.push_back(i);

            std::vector<int> source(input.size());
            std::fill(source.begin(), source.end(), -1);
            layers[i]->add_input(input, source);
        }
        pending.push_back(in_edges[i].size());
    }

    while (!ready.empty()) {
        int layer_id = ready[0];
        ready.erase(ready.begin());
        std::vector<Eigen::MatrixXd> out = layers[layer_id]->compute(target);
        std::vector<int> dests = out_edges[layer_id];

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

    std::vector<std::pair<int, int>> order = scheduleExecute();

    std::vector<std::vector<Eigen::MatrixXd>> grads(layers.size());

    for(int i = 0; i < order.size(); i++){
        std::pair<int, int> execute = order[i];
        int layer = execute.second;
        if (execute.first == -1){
            grads[layer] = layers[layer]->grad(std::vector<Eigen::MatrixXd>(), lr);
        } else {
            grads[layer] = layers[layer]->grad(grads[execute.first], lr);
        }
    }
}

std::vector<std::pair<int, int>> Schedular::scheduleExecute() {
    std::vector<int> ready;
    std::vector<int> pending(layers.size());

    // the gradient compute order,
    // first is father, -1 means no father;
    // second is current compute node
    std::vector<std::pair<int, int>> order;

    for(int i = 0; i < layers.size(); i++){
        if (out_edges[i].size() == 0){
            ready.push_back(i);
            order.push_back(std::make_pair(-1, i));
        }
        pending[i] = out_edges[i].size();
    }

    while (!ready.empty()){
        int l = ready.back();
        ready.pop_back();

        for(int i = 0; i < in_edges[l].size(); i++){
            int next = in_edges[l][i];
            order.push_back(std::make_pair(l, next));
            pending[next] -= 1;

            if (pending[next] == 0){
                ready.push_back(next);
            }
        }
    }

    return order;
}