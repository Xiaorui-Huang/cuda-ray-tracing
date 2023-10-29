#include <stdio.h>

class Base {
  public:
    __device__ virtual void print() { printf("Base\n"); }
};

class Derived : public Base {
  public:
    __device__ void print() override { printf("Derived\n"); }
};

// virtual functions can only be called in __global__ if created on device
__global__ void kernel() {
    Derived *d = new Derived();
    Base *b = d;
    b->print(); // Should print "Derived"
    delete d;
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
