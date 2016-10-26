#ifndef DCDM_H_
#define DCDM_H_

#include "Dual_solver.hpp"
// #include "Tool.hpp"
#include <fmath.hpp>

namespace sdm {

class Dcdm : public Dual_solver {
public:
  Dcdm(const std::string &train_libsvm_format, const double &c = 1.0,
       const double &gamma = 0.01, const double &stpctr = 1e-128,
       const int &max_iter = 300);
  Dcdm(const std::string &train_libsvm_format,
       const std::string &valid_libsvm_format, const double &c = 1.0,
       const double &gamma = 1.0, const double &stpctr = 1e-128,
       const int &max_iter = 300);
  ~Dcdm();

  void set_regularized_parameter(const double &c);
  void set_kernel_parameter(const double &gmm);

  int get_train_l(void);
  int get_valid_l(void);

  double predict(const Eigen::ArrayXd &alpha) const;
  Eigen::ArrayXd train_warm_start(const Eigen::ArrayXd &alpha)
  ;

  double get_dual_func(const Eigen::ArrayXd &alpha);
  double get_primal_func(const Eigen::ArrayXd &alpha);
  double get_grad_norm(const Eigen::ArrayXd &alpha);

  std::vector<double> get_c_set_right_opt(const Eigen::ArrayXd &alpha_opt,
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
  double gamma;
  double stopping_criterion;
  int max_iteration;

  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x_;
  Eigen::ArrayXd valid_y_;
  int valid_l_;
  int valid_n_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 1> kernel_valid_;

  Eigen::VectorXd grad_;
  Eigen::VectorXd wx_;

  void set_rbf_kernel(void);
  void set_rbf_kernel_valid(void);

  void update_grad(const double &delta, const int &index);
  void set_wx(const Eigen::ArrayXd &alpha);
  double get_w_squared_norm(const Eigen::ArrayXd &alpha);

};
} // namespace sdm

#endif
