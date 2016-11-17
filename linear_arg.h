//
// Created by yixuan he on 17/11/2016.
//

#ifndef DL_LINEAR_ARG_H
#define DL_LINEAR_ARG_H

#include <vector>
#include <iterator>

class Matrix{
public:
    Matrix(std::vector<std::vector<double>> data): value(data){}
    Matrix(int** data, int shape[]) {
        value = std::vector<std::vector<double>>();
        int row = shape[0], colunm = shape[1];
        for(int i = 0; i < row; i++){
            std::vector<double> d = std::vector<double>(data[i], data[i] + colunm);
            value.push_back(d);
        }
    }

    void get_shape(int shape[]){
        shape[0] = value.size();
        shape[1] = value[0].size();
    }

    Vector mul(Vector v, int* err){

    }

    Matrix mul(Matrix m, int *err){

    }

private:
    std::vector<std::vector<double>> value;
};

class Vector{
public:
    std::vector<double> getValue() const{
        return value;
    }

    Vector(double data[], int size) {
        value = std::vector<double>(data, data + size);
    }
    Vector(std::vector<double> data) : value(data){}

    int get_size() const {
        return value.size();
    }

    double mul(const Vector& v, int* err) {
        if (this->get_size() != v.get_size()){
            *err = 1;
            return 0;
        }

        double result = 0;
        std::vector<double> other = v.getValue();

        for(int i = 0; i < v.get_size(); i++){
            result += value[i]*other[i];
        }

        *err = 0;
        return result;
    }

    Vector mul(const Matrix& m, int* err){

    }
private:
    std::vector<double> value;
};


#endif //DL_LINEAR_ARG_H
