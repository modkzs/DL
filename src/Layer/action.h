//
// Created by yixuan he on 18/11/2016.
//

#ifndef DL_ACTION_H
#define DL_ACTION_H
#include <map>
#include <iostream>

# define REGISTER_ACTIVE_FUNC(name, fn) \
    bool _active_result_##fn = active_register(name, fn)

# define REGISTER_ACTIVE_GRAD_FUNC(name, fn) \
    bool _active_result_##fn = active_grad_register(name, fn)


typedef double (*active)(double x);
typedef double (*active_grad)(double x);

static std::map<std::string, active> _active_map;
static std::map<std::string, active_grad> _active_grad_map;

bool active_register(std::string name, active func){
    auto ret = _active_map.insert(std::pair<std::string,active>(name, func));
    return ret.second;
}
bool active_grad_register(std::string name, active_grad func){
    auto ret = _active_grad_map.insert(std::pair<std::string,active_grad>(name, func));
    return ret.second;
}

double sigmod(double x){
    return 1/(1 + std::exp(-x));
}
REGISTER_ACTIVE_FUNC("sigmod", sigmod);

double sigmod_grad(double x){
    double output = 1/(1 + std::exp(-x));
    return output*(1-output);
}
REGISTER_ACTIVE_GRAD_FUNC("sigmod", sigmod_grad);

double line(double x){
    return x;
}
REGISTER_ACTIVE_FUNC("line", line);

double  line_grad(double x){
    return 1;
}
REGISTER_ACTIVE_GRAD_FUNC("line", line_grad);

#endif //DL_ACTION_H
