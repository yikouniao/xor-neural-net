/* This is a BP net model which has only one hiden layer.
 * The numbers of input/hiden/output nodes are defined in sample.h
 * Activation function: Sigmod function f(x) = 1 / (1 + e^(-x))
 * Hiden layer:
 *   layer 1
 *     nodes number: 2 * input_number + 1
 *     learning rate: fixed as 0.6
 *     threshold: fixed as 0
 * Output layer:
 *   learning rate: fixed as 0.6
 *   threshold: fixed as 0
 * Convergence checking:
 *   |error| < 0.008 */

#ifndef _BPNET_H
#define _BPNET_H

#include "sample.h"

#define HIDEN ((2 * IN) + 1)

class BpNet {
 public:
  BpNet(double rate_h1_ = 0.6, double rate_o_ = 0.6, double err_thres_ = 0.008);
  ~BpNet();

 private:
  std::array<std::array<double, HIDEN>, IN> w_h1; // the input weights of hiden nodes in 1st layer
  std::array<std::array<double, OUT>, HIDEN> w_o; // the input weights of output nodes
  std::array<double, HIDEN> thres_h1; // threshold of hiden nodes in 1st layer
  std::array<double, OUT> thres_o; // threshold of output nodes
  double rate_h1; // learning rate of hiden nodes in 1st layer
  double rate_o; // learning rate of output nodes
  double err_thres; // threshold of convergence checking
};

#endif
