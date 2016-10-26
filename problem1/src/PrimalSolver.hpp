#ifndef PRIMAL_SOLVER_HPP_
#define PRIMAL_SOLVER_HPP_

#include "Tool_impl.hpp"
#include <algorithm>

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace sdm{

class PrimalSolver {
public:
  virtual ~PrimalSolver(void) {}
  virtual int get_valid_l(void) = 0;
  virtual int get_valid_n(void) = 0;
  virtual Eigen::VectorXd train_warm_start(const Eigen::VectorXd &w) = 0;

  // virtual Eigen::VectorXd train_warm_start_inexact(const Eigen::VectorXd &w,
  //                                                  const double
  //                                                  inexact_level,
  //                                                  double &ub_validerr,
  //                                                  double &lb_validerr) = 0;

};

}
#endif //PrimalSolver_hpp_
