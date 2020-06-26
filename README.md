# alpaka_strided_copy
Minimal working example of a strided copy from host to device memory utiliying the alpaka library.

In combination with the NVidia profiler, this reveals the copy speed from host to device memory if the data is contiguos in memory but only every other chunk should be copied. 

## Requirements

* [alpaka](https://github.com/alpaka-group/alpaka) 0.4 or higher
* Boost 1.65.1 or higher (dependency of alpaka)
* CUDA 9 or higher 

## Build

```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Run with: ```./alpaka_strided_copy```
