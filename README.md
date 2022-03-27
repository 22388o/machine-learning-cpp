# machine-learning

### Running some program here will require matplotlib-cpp library


##### To run a programm that requires [matplotlib-cpp](https://github.com/lava/matplotlib-cpp) just execute this command or read more documentations [here](https://github.com/lava/matplotlib-cpp).

```bash
$ g++ main.cpp \
  -std=c++14 -I/usr/include/python3.9 \
  -I/usr/lib/python3.9/site-packages/numpy/core/include \
  -lpython3.9 \
  -o main 

$ ./main
```
