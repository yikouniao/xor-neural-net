#ifndef _SAMPLE_H
#define _SAMPLE_H

#include <array>
#include <vector>

#define IN 2
#define OUT 1

using array_i = std::array<double, IN>;
using array_o = std::array<double, OUT>;

struct Data {
  array_i in;
  array_o out;
};

extern const std::vector<Data> samples;

#endif
