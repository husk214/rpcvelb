#ifndef PRIMAL_SOLVER_HPP_
#define PRIMAL_SOLVER_HPP_

#include "Solver.hpp"
#include "Primal_function.hpp"
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace sdm {
class Primal_solver {
public:
  virtual ~Primal_solver() {}
  virtual void set_regularized_parameter(const double &c) = 0;
  virtual int get_train_n() = 0;
  virtual int get_valid_l() = 0;
  virtual int get_valid_n() = 0;
  virtual Eigen::VectorXd get_grad() = 0;

  // virtual Eigen::VectorXd train() = 0;
  // virtual Eigen::VectorXd train_warm_start(const Eigen::VectorXd &alpha) = 0;

  virtual Eigen::VectorXd train_warm_start_inexact(const Eigen::VectorXd &w,
                                                   const double inexact_level,
                                                   int &num_ub,
                                                   int &num_lb) = 0;

  virtual std::vector<double>
  get_c_set_right_subopt(const bool &flag_sort = true) const = 0;
  virtual std::vector<double>
  get_c_set_left_subopt(const bool &flag_sort = true) const = 0;

  virtual std::vector<double>
  get_c_set_right_subopt(const double &c_now, const Eigen::VectorXd &w,
                         const Eigen::VectorXd &grad_w,
                         const bool &flag_sort = true) const = 0;
  virtual std::vector<double>
  get_c_set_left_subopt(const double &c_now, const Eigen::VectorXd &w,
                        const Eigen::VectorXd &grad_w,
                        const bool &flag_sort = true) const = 0;

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
