#include "bpnet.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

BpNet::BpNet(double rate_h1_, double rate_o_, double err_thres_)
    : rate_h1(rate_h1_), rate_o(rate_o_), err_thres(err_thres_) {
  srand(time(0)); // use current time as seed for random generator

  // initialize w_h1
  for (auto& w_each_in : w_h1) {
    for (auto& w : w_each_in) {
      w = (double)rand() / RAND_MAX;
    }
  }

  // initialize w_o
  for (auto& w_each_hiden : w_o) {
    for (auto& w : w_each_hiden) {
      w = (double)rand() / RAND_MAX;
    }
  }

  // initialize thres_h1
  thres_h1 = 0;

  // initialize thres_o
  thres_o = 0;
}

BpNet::~BpNet() {}
