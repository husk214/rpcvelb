#ifndef PRIMAL_FUNCTION_HPP_
#define PRIMAL_FUNCTION_HPP_

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace sdm {

class Primal_function {
public:
  virtual ~Primal_function(void) {}

  virtual double get_func(const Eigen::VectorXd &w) = 0;
  virtual Eigen::VectorXd get_grad(const Eigen::VectorXd &w) = 0;
  virtual Eigen::VectorXd get_loss_grad(const Eigen::VectorXd &w) = 0;
  virtual Eigen::VectorXd product_hesse_vec(const Eigen::VectorXd &V) = 0;

  // get number of features
  virtual int get_variable(void) = 0;
  virtual double get_regularized_parameter(void) = 0;
  virtual void set_regularized_parameter(const double &c) = 0;
};

} // namespace sdm

#endif //PRIMAL_FUNCTION_HPP_
