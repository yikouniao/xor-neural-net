#include "bpnet.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

BpNet::BpNet(double rate_w_h1_, double rate_w_o_, double rate_thres_h1_,
             double rate_thres_o_, double err_thres_)
    : rate_w_h1(rate_w_h1_), rate_w_o(rate_w_o_), rate_thres_h1(rate_thres_h1_),
      rate_thres_o(rate_thres_o_), err_thres(err_thres_) {
  srand(time(0)); // use current time as seed for random generator

  // initialize w_h1
  for (auto& w_each_in : w_h1) {
    for (auto& w : w_each_in) {
      w = GetRand();
    }
  }

  // initialize w_o
  for (auto& w_each_hiden : w_o) {
    for (auto& w : w_each_hiden) {
      w = GetRand();
    }
  }

  // initialize thres_h1
  for (auto& thres_each_hiden : thres_h1) {
    thres_each_hiden = GetRand();
  }

  // initialize thres_o
  for (auto& thres_each_o : thres_o) {
    thres_each_o = GetRand();
  }
}

BpNet::~BpNet() {}

void BpNet::Train() {
  const int samples_num = samples_in.size();
  int train_times = 0;
  bool conv = FALSE;
  while (!conv) {
    conv = TRUE;
    ++train_times;
    cout << "The " + train_times + " times training...\n";
    for (int samples_order = 0; samples_order < samples_num; ++samples_order) {
      // Propagation
      array<double, HIDEN> out_h1 = {0};
      GetOutH1(samples_order, out_h1);

      array<double, OUT> out_out = {0};
      GetOutOut(samples_order, out_out);

      array<double, OUT> err;
      GetErr(samples_order, err);
      if (!CheckConv(err))
        conv = FALSE;
      cout << "Sample " + samples_order + " error: ";
      for (const auto& e : err) {
        cout << e << " ";
      }
      cout << "\n";

      // weight update

    }
  }
}

void BpNet::GetOutH1(int samples_order, array<double, HIDEN>& out_h1) {
  for (int i = 0; i < HIDEN; ++i) {
    for (int j = 0; j < IN; ++j) {
      out_h1[i] += samples_in[samples_order][j] * w_h1[j][i];
    }
    out_h1[i] += thres_h1[i];
    out_h1[i] = Sigmod(out_h1[i]);
  }
}

void BpNet::GetOutOut(int samples_order, array<double, OUT>& out_out) {
  for (int i = 0; i < OUT; ++i) {
    for (int j = 0; j < HIDEN; ++j) {
      out_out[i] += out_h1[samples_order][j] * w_o[j][i];
    }
    out_out[i] += thres_o[i];
    out_out[i] = Sigmod(out_out[i]);
  }
}

void BpNet::GetErr(int samples_order, array<double, OUT>& err) {
  for (int i = 0; i < OUT; ++i) {
    err[i] = samples_out[samples_order][i] - out_out[i];
    if (err[i] < 0)
      err[i] = -err[i];
  }
}

bool BpNet::CheckConv(const array<double, OUT>& err) {
  for (const auto& e : err) {
    if (e < err_thres) {
      continue;
    } else {
      return FALSE;
    }
  }
  return TRUE;
}

inline double GetRand() {
  return (double)rand() / RAND_MAX - 0.5;
}

inline double Sigmod(double x) {
  return 1 / (1 + exp(-x));
}
