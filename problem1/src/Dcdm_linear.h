#ifndef DCDM_LINEAR_H_
#define DCDM_LINEAR_H_

#include "PrimalSolver.hpp"
// #include "Tool.hpp"
#include <fmath.hpp>

namespace sdm {

class Dcdm_linear : virtual public PrimalSolver {
public:
  Dcdm_linear(const std::string &train_libsvm_format, const double &c = 1.0,
               const double &stpctr = 1e-128,
              const int &max_iter = 1000);
  Dcdm_linear(const std::string &train_libsvm_format,
              const std::string &valid_libsvm_format, const double &c = 1.0,
               const double &stpctr = 1e-128,
              const int &max_iter = 300);
  ~Dcdm_linear();

  void set_regularized_parameter(const double &c);

  int get_train_l(void);
  int get_valid_l(void);
  int get_valid_n(void);
  double predict(const Eigen::VectorXd &w) const;
  Eigen::VectorXd train_warm_start(const Eigen::VectorXd &alpha);

  double get_dual_func(const Eigen::VectorXd &alpha);
  double get_primal_func(const Eigen::VectorXd &w);
  Eigen::VectorXd get_grad(const Eigen::VectorXd &w);
  double get_grad_norm(const Eigen::VectorXd &w);

  std::vector<double> get_c_set_right_opt(const Eigen::VectorXd &w,
                                          const double &c_now,
                                          double &valid_err);
  // Eigen::VectorXd train_warm_start_inexact(const Eigen::VectorXd &alpha,
  //                                          const double inexact_level,
  //                                          double &ub_validerr,
  //                                          double &lb_validerr);

private:
  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> train_x_;
  Eigen::ArrayXd train_y_;
  int train_l_;
  int train_n_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1> kernel_;

  double C;
  double stopping_criterion;
  int max_iteration;

  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x_;
  Eigen::ArrayXd valid_y_;
  int valid_l_;
  int valid_n_;

  Eigen::VectorXd w_;
  Eigen::VectorXd grad_;
  Eigen::VectorXd wx_;
  Eigen::VectorXd x_sqnorm_;

  void set_Q(void);

  void update_grad(const double &delta, const unsigned int &index);
};
} // namespace sdm

#endif
