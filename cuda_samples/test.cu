#include <cstdio>
#include <tuple>

using Point = std::tuple<float, float, float>;

__global__ void addPoints(const Point *a, const Point *b, Point *result) {
    int idx = threadIdx.x;
    if (idx == 0) {
        std::get<0>(*result) = std::get<0>(*a) + std::get<0>(*b);
        std::get<1>(*result) = std::get<1>(*a) + std::get<1>(*b);
        std::get<2>(*result) = std::get<2>(*a) + std::get<2>(*b);
    }
}

int main() {
    Point h_a(1.0f, 2.0f, 3.0f);
    Point h_b(4.0f, 5.0f, 6.0f);
    Point h_result;

    Point *d_a, *d_b, *d_result;

    cudaMalloc(&d_a, sizeof(Point));
    cudaMalloc(&d_b, sizeof(Point));
    cudaMalloc(&d_result, sizeof(Point));

    cudaMemcpy(d_a, &h_a, sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(Point), cudaMemcpyHostToDevice);

    addPoints<<<1, 1>>>(d_a, d_b, d_result);

    cudaMemcpy(&h_result, d_result, sizeof(Point), cudaMemcpyDeviceToHost);

    printf("Result: (%f, %f, %f)\n", std::get<0>(h_result), std::get<1>(h_result), std::get<2>(h_result));

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    return 0;
}
