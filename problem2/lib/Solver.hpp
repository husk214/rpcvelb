#ifndef Solver_HPP_
#define Solver_HPP_

#include "Function.hpp"
#include <vector>

class Solver {
public:
  virtual ~Solver(void) {}

  virtual void set_fun_obj(Function *fun_o) = 0;
  virtual Function *get_fun_obj(void) = 0;
  virtual Eigen::VectorXd get_w(void) = 0;
  virtual Eigen::VectorXd get_grad(void) = 0;
  virtual Eigen::VectorXd get_loss_grad(const Eigen::VectorXd &w) = 0;
  virtual double get_grad_norm(void) = 0;
  virtual Eigen::VectorXd train_warm_start(const Eigen::VectorXd &w) = 0;

  virtual Eigen::VectorXd train_warm_start_inexact(const Eigen::VectorXd &w,
                                                   const double inexact_level,
                                                   double &ub_validerr,
                                                   double &lb_validerr) = 0;
  virtual int get_valid_l(void) = 0;
  virtual int get_valid_n(void) = 0;
  virtual double get_valid_error(const Eigen::VectorXd &w) = 0;
  virtual void get_upper_lower_bound_valid_error(double &ub_ve,
                                                 double &lb_ve) = 0;
  virtual double get_lower_bound_valid_error(const Eigen::VectorXd &w,
                                             const Eigen::VectorXd &grad,
                                             const double c_now,
                                             const double c) = 0;

  virtual void get_ub_lb_ve_apprx(double &ub_ve, double &lb_ve) = 0;
  virtual double get_lb_ve_apprx(const Eigen::VectorXd &w_tilde,
                                 const Eigen::VectorXd &grad,
                                 const double c_now, const double c) = 0;

  virtual std::vector<double> get_c_set_right_opt(const Eigen::VectorXd &w_star,
                                                  const double &c_now) = 0;

  virtual std::vector<double>
  get_c_set_right_subopt(const Eigen::VectorXd &w,
                         const Eigen::VectorXd &grad_w,
                         const double &c_now) = 0;
  virtual std::vector<double>
  get_c_set_left_subopt(const Eigen::VectorXd &w, const Eigen::VectorXd &grad_w,
                        const double &c_now) = 0;

  virtual std::vector<double>
  get_c_set_right_opt_for_path(const Eigen::VectorXd &w,
                               const double &c_now) = 0;
  virtual std::vector<double>
  get_c_set_right_subopt_for_path(const Eigen::VectorXd &w,
                                  const Eigen::VectorXd &grad_w,
                                  const double &c_now, int &num_dif_ublb) = 0;
};

class Solver_CV {
public:
  virtual ~Solver_CV(void) {}

  // virtual Eigen::VectorXd get_grad(void) = 0;
  // virtual double get_grad_norm(void) = 0;

  virtual std::vector<Eigen::VectorXd>
  train_inexact_cv(const std::vector<Eigen::VectorXd> &w_set,
                   const double inexact_level, const int fold_num) = 0;
};

#endif // Solver_HPP
