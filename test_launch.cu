__global__ void my_kernel() {}
int main() {
    auto k = my_kernel;
    k<<<1, 1>>>();
    return 0;
}
