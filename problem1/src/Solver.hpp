#ifndef SOLVER_HPP_
#define SOLVER_HPP_

#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace sdm {
class Solver {
public:
  virtual ~Solver(void) {}

  // virtual Eigen::VectorXd train() = 0;
  virtual void set_regularized_parameter(const double &c) = 0;
  virtual int get_train_l(void) = 0;
  virtual int get_valid_l(void) = 0;
  virtual Eigen::ArrayXd train_warm_start(const Eigen::ArrayXd &alpha) = 0;
  virtual Eigen::VectorXd train_warm_start(const Eigen::VectorXd &alpha) = 0;

  virtual std::vector<double>
  get_c_set_right_opt(const Eigen::ArrayXd &alpha_opt, const double &c_now,
                      double &valid_err) = 0;

  // virtual Eigen::VectorXd train_warm_start_inexact(const Eigen::VectorXd
  // &alpha,
  //                                                  const double
  //                                                  inexact_level,
  //                                                  double &ub_validerr,
  //                                                  double &lb_validerr) = 0;

  // virtual std::vector<double>
  // get_c_set_right_opt(const Eigen::ArrayXd &alpha_opt, const double &c_now,
  //                     double &valid_err) = 0;

  // virtual std::vector<double>
  // get_c_set_right_subopt(const Eigen::VectorXd &w,
  //                        const Eigen::VectorXd &grad_w,
  //                        const double &c_now) = 0;

  // virtual std::vector<double>
  // get_c_set_left_subopt(const Eigen::VectorXd &w, const Eigen::VectorXd
  // &grad_w,
  //                       const double &c_now) = 0;
};
} // namespace sdm

#endif // PRIMAL_SOLVER_HPP_
