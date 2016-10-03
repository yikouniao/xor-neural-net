/* This is a BP net model which has only one hiden layer.
 * The numbers of input/hiden/output nodes are defined in sample.h
 * Activation function: Sigmod function f(x) = 1 / (1 + e^(-x))
 * Hiden layer:
 *   layer 1
 *     nodes number: 2 (It should be about 2 * input_num + 1, but XOR is a simple problem)
 *     learning rate of weight value: fixed as 0.6
 *     learning rate of threshold: fixed as 0.6
 * Output layer:
 *   learning rate of weight value: fixed as 0.6
 *   learning rate of threshold: fixed as 0.6
 * Convergence checking:
 *   |error| < 0.008 */

#ifndef _BPNET_H
#define _BPNET_H

#include "sample.h"

#define HIDEN 2 // ((2 * IN) + 1)

class BpNet {
 public:
  BpNet(double rate_w_h1_ = 0.6, double rate_w_o_ = 0.6,
        double rate_thres_h1_ = 0.6, double rate_thres_o_ = 0.6,
        double err_thres_ = 0.008);
  ~BpNet();

  // train the neural net
  void Train();

 private:
  std::array<std::array<double, HIDEN>, IN> w_h1; // the input weights of hiden nodes in 1st layer
  std::array<std::array<double, OUT>, HIDEN> w_o; // the input weights of output nodes
  std::array<double, HIDEN> thres_h1; // threshold of hiden nodes in 1st layer
  std::array<double, OUT> thres_o; // threshold of output nodes
  double rate_w_h1; // learning rate of hiden nodes in 1st layer
  double rate_w_o; // learning rate of output nodes
  double rate_thres_h1; // learning rate of hiden nodes in 1st layer
  double rate_thres_o; // learning rate of output nodes
  double err_thres; // threshold of convergence checking

  // compute the output of hiden nodes in 1st layer
  void GetOutH1(int samples_order, array<double, HIDEN>& out_h1);

  // compute the output of output nodes
  void GetOutO(int samples_order, array<double, OUT>& out_o);

  // compute the errors of output nodes
  void GetErrO(int samples_order, const array<double, OUT>& out_o, array<double, OUT>& err_o);

  // check weither the errors are acceptable
  bool CheckConv(const array<double, OUT>& err_o);

  // compute the errors of hiden nodes in 1st layer
  void GetErrH1();
};

#endif
