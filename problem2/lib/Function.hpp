#ifndef Function_HPP_
#define Function_HPP_

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include <string>

class Function {
public:
  virtual ~Function(void) {}
  virtual Function *create(const std::string fileNameLibSVMFormat) const = 0;

  virtual double get_func(const Eigen::VectorXd &w) = 0;
  virtual Eigen::VectorXd get_grad(const Eigen::VectorXd &w) = 0;
  virtual Eigen::VectorXd get_loss_grad(const Eigen::VectorXd &w) = 0;
  virtual Eigen::VectorXd product_hesse_vec(const Eigen::VectorXd &V) = 0;

  // get number of features
  virtual int get_variable(void) = 0;
  virtual double get_regularized_parameter(void) = 0;
  virtual void set_regularized_parameter(const double &c) = 0;

  // predict Accuracy using w , vX and vy is testdate
  virtual double predict(const std::string fileNameLibSVMFormat,
                         Eigen::VectorXd &w) = 0;
};

class Function_CV {
public:
  virtual ~Function_CV(void) {}
  virtual Function_CV *create(const std::string fileNameLibSVMFormat) const = 0;

  virtual double get_func(const Eigen::VectorXd &w, const int fold_i) = 0;
  virtual Eigen::VectorXd get_grad(const Eigen::VectorXd &w, const int fold_i) = 0;
  virtual Eigen::VectorXd product_hesse_vec(const Eigen::VectorXd &V, const int fold_i) = 0;

  // get number of features
  virtual int get_variable(void) = 0;
  virtual double get_regularized_parameter(void) = 0;
  virtual void set_regularized_parameter(const double &c) = 0;

  // predict Accuracy using w , vX and vy is testdate
  virtual double predict(const std::string fileNameLibSVMFormat,
                         Eigen::VectorXd &w) = 0;
};

#endif
