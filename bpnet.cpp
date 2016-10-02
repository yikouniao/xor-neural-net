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

void BpNet::GetOutH1(int samples_order, array<double, HIDEN>& out_h1) {
  for (int i = 0; i < HIDEN; ++i) {
    for (int j = 0; j < IN; ++j) {
      out_h1[i] += samples_in[samples_order][j] * w_h1[j][i];
    }
    out_h1[i] += thres_h1[i];
    out_h1[i] = 1 / (1 + exp(-out_h1[i]));
  }
}

void BpNet::GetOutOut(int samples_order, array<double, OUT>& out_out) {
  for (int i = 0; i < OUT; ++i) {
    for (int j = 0; j < HIDEN; ++j) {
      out_out[i] += out_h1[samples_order][j] * w_o[j][i];
    }
    out_out[i] += thres_o[i];
    out_out[i] = 1 / (1 + exp(-out_out[i]));
  }
}

void BpNet::GetErr(int samples_order, array<double, OUT>& err) {
  for (int i = 0; i < OUT; ++i) {
    err[i] = samples_out[samples_order][i] - out_out[i];
    if (err[i] < 0)
      err[i] = -err[i];
  }
}

void BpNet::Train() {
  int conv_num = 0; // numbers of convergent samples
  int samples_num = samples_in.size();
  while (conv_num < samples_num) { // while not every sample converges
    for (int i = 0; i < samples_num; ++i) { // train every sample
      array<double, HIDEN> out_h1 = {0};
      GetOutH1(samples_order, out_h1);

      array<double, OUT> out_out = {0};
      GetOutOut(samples_order, out_out);

      array<double, OUT> err;
      GetErr(samples_order, err);
    }
  }
}
