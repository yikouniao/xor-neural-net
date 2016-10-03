#include "bpnet.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

namespace {
// get a random number in [-0.1 0.1]
inline double GetRand() {
  return 0.2 * rand() / RAND_MAX - 0.1;
}

// sigmod function
inline double Sigmod(double x) {
  return 1 / (1 + exp(-x));
}
} // namespace

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
  const int samples_num = samples.size();
  int train_times = 0;
  bool conv = false;
  while (!conv) {
    conv = true;
    ++train_times;
    cout << "The " << train_times << " times training...\n";
    for (int samples_order = 0; samples_order < samples_num; ++samples_order) {
      // Propagation
      array_h1 out_h1 = {0};
      GetOutH1(samples_order, out_h1);

      array_o out_o = {0};
      GetOutO(out_h1, out_o);

      array_o err_o;
      GetErrO(samples_order, out_o, err_o);
      if (!CheckConv(err_o))
        conv = false;
      cout << "Sample " << samples_order << " error: ";
      for (const auto& e : err_o) {
        cout << e << " ";
      }
      cout << "\n";

      // weight update

    }
  }
}

void BpNet::GetOutH1(int samples_order, array_h1& out_h1) {
  for (int i = 0; i < HIDEN1; ++i) {
    for (int j = 0; j < IN; ++j) {
      out_h1[i] += samples[samples_order].in[j] * w_h1[j][i];
    }
    out_h1[i] += thres_h1[i];
    out_h1[i] = Sigmod(out_h1[i]);
  }
}

void BpNet::GetOutO(const array_h1& out_h1, array_o& out_o) {
  for (int i = 0; i < OUT; ++i) {
    for (int j = 0; j < HIDEN1; ++j) {
      out_o[i] += out_h1[j] * w_o[j][i];
    }
    out_o[i] += thres_o[i];
    out_o[i] = Sigmod(out_o[i]);
  }
}

void BpNet::GetErrO(int samples_order, const array_o& out_o, array_o& err_o) {
  for (int i = 0; i < OUT; ++i) {
    err_o[i] = samples[samples_order].out[i] - out_o[i];
  }
}

bool BpNet::CheckConv(const array_o& err_o) {
  for (const auto& e : err_o) {
    if (e < err_thres && e > -err_thres) {
      continue;
    } else {
      return false;
    }
  }
  return true;
}

void BpNet::GetErrH1(const array_o& err_o, array_h1& err_h1) {

}
