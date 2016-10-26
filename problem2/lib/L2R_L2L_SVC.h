#ifndef L2R_L2L_SVC_
#define L2R_L2L_SVC_

#include "Function.hpp"
#include "Tools.h"
#include <string>

class L2R_L2L_SVC : public Function {
public:
  L2R_L2L_SVC(const std::string fileNameLibSVMFormat, const double c = 1.0);
  ~L2R_L2L_SVC();
  L2R_L2L_SVC *create(const std::string fileNameLibSVMFormat) const;

  double get_func(const Eigen::VectorXd &w);
  Eigen::VectorXd get_grad(const Eigen::VectorXd &w);
  Eigen::VectorXd get_loss_grad(const Eigen::VectorXd &w);
  Eigen::VectorXd product_hesse_vec(const Eigen::VectorXd &V);

  int get_variable(void);
  double get_regularized_parameter(void);
  void set_regularized_parameter(const double &c);

  double predict(const std::string fileNameLibSVMFormat, Eigen::VectorXd& w);

protected:
  // ----- members -----
  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> X; // patarn matrix , xi =
  Eigen::ArrayXd y;    // indicator vector y , yi={+1,-1}
  Eigen::ArrayXd z;    // z = C(X_{I,:}w -yi)
  std::vector<int> vI; // vI element is i , I = {i | 1-yi<w,xi> > 0}
  int l;               // of data
  int n;               // of features
  double C; // penalty parameter

  // return product between X_{I,:} and vector V
  Eigen::VectorXd productXVsub(const Eigen::VectorXd &V);

  // return product between X_{I,:}^t and vector V
  Eigen::VectorXd productXtVsub(const Eigen::VectorXd &V);
};

#endif
