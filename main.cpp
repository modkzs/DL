#include <iostream>
#include "linear_arg.h"

int main() {
    double a[] = {1,2,3,4,5};
    Vector v(a, sizeof(a)/ sizeof(a[0]));
    int err;
    std::cout << "Mul : " << v.mul(Vector(a, sizeof(a)/ sizeof(a[0])), &err) << std::endl;

    std::cout << "Hello, World!" << std::endl;
    return 0;
}