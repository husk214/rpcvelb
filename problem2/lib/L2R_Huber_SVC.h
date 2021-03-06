#ifndef L2R_HUBER_SVC_H_
#define L2R_HUBER_SVC_H_

#include "Function.hpp"
#include "Tools.h"
#include <string>

class L2R_Huber_SVC : public Function {
public:
  L2R_Huber_SVC(const std::string fileNameLibSVMFormat, const double c = 1.0);
  L2R_Huber_SVC(const Eigen::SparseMatrix<double, 1, std::ptrdiff_t> train_x,
                const Eigen::ArrayXd train_y, const double c = 1.0);

  ~L2R_Huber_SVC();
  L2R_Huber_SVC *create(const std::string fileNameLibSVMFormat) const;

  double get_func(const Eigen::VectorXd &w);
  Eigen::VectorXd get_grad(const Eigen::VectorXd &w);
  Eigen::VectorXd get_loss_grad(const Eigen::VectorXd &w);
  Eigen::VectorXd product_hesse_vec(const Eigen::VectorXd &V);

  int get_variable(void);
  double get_regularized_parameter(void);
  void set_regularized_parameter(const double &c);

  double predict(const std::string fileNameLibSVMFormat, Eigen::VectorXd &w);

private:
  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> X;
  Eigen::ArrayXd y;
  int l;
  int n;

  double C;

  Eigen::ArrayXd z;
  std::vector<int> indexs_D1;
  std::vector<int> indexs_D2;
  Eigen::VectorXd XD2w_yD2;

  Eigen::VectorXd productXVsub2(const Eigen::VectorXd &V);
  Eigen::VectorXd productXtVsub1(const Eigen::VectorXd &V);
  Eigen::VectorXd productXtVsub2(const Eigen::VectorXd &V);
};

#endif
