#ifndef L2R_LR_H_
#define L2R_LR_H_

#include "Function.hpp"
#include "Tools.h"
#include <string>

class L2R_LR : public Function {
public:
  L2R_LR(const std::string fileNameLibSVMFormat, const double c = 1.0);
  ~L2R_LR();

  L2R_LR *create(const std::string fileNameLibSVMFormat) const;

  double get_func(const Eigen::VectorXd &w);
  Eigen::VectorXd get_grad(const Eigen::VectorXd &w);
  Eigen::VectorXd get_loss_grad(const Eigen::VectorXd &w);
  Eigen::VectorXd product_hesse_vec(const Eigen::VectorXd &V);

  int get_variable(void);
  double get_regularized_parameter(void);
  void set_regularized_parameter(const double &c);
  double predict(const std::string fileNameLibSVMFormat, Eigen::VectorXd &w);

protected:
  // ----- members -----
  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> X; // patarn matrix , xi =
  Eigen::ArrayXd y; // indicator vector y , yi={+1,-1}
  Eigen::ArrayXd z;
  Eigen::ArrayXd D;
  int l; // of data
  int n; // of features
  // double bias;    // bais term, if (bias =< 0) then no bias term
  double C; // penalty parameter
};

#endif
