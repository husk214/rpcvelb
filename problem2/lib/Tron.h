#ifndef TRON_H_
#define TRON_H_

#include <iostream>
#include <string>
#include <iomanip>
#include <ctime>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Eigen/LU>

#include <chrono>
#include "Solver.hpp"
#include "Tools.h"

class Tron : public Solver {
public:
  Tron(Function *fun_obj, const double eps = 1e-6, const int max_iter = 1000);
  Tron(Function *fun_obj, const std::string valid_file, const double eps = 1e-6,
       const int max_iter = 1000);
  Tron(Function *fun_obj,
       Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x,
       Eigen::ArrayXd valid_y, const double eps = 1e-6,
       const int max_iter = 1000);
  ~Tron();

  Function *get_fun_obj();
  int get_max_iter();
  double get_eps();
  Eigen::VectorXd get_grad();
  Eigen::VectorXd get_loss_grad(const Eigen::VectorXd &w);
  double get_grad_norm();

  Eigen::VectorXd tron();
  Eigen::VectorXd tron_cg(const double delta, const Eigen::VectorXd &g,
                          Eigen::VectorXd &r, int &cg_iter, double gnorm);

  void set_fun_obj(Function *fun_o);
  Eigen::VectorXd train_warm_start(const Eigen::VectorXd &w_start);
  Eigen::VectorXd train_warm_start_inexact(const Eigen::VectorXd &w,
                                           const double inexact_level,
                                           double &ub_validerr,
                                           double &lb_validerr);

  Eigen::VectorXd get_w();
  int get_valid_l();
  int get_valid_n();
  double get_valid_error(const Eigen::VectorXd &w);
  void get_upper_lower_bound_valid_error(double &ub_ve, double &lb_ve);
  double get_lower_bound_valid_error(const Eigen::VectorXd &w,
                                     const Eigen::VectorXd &grad,
                                     const double c_now, const double c);

  void get_ub_lb_ve_apprx(double &ub_ve, double &lb_ve);
  double get_lb_ve_apprx(const Eigen::VectorXd &w_tilde,
                         const Eigen::VectorXd &grad, const double c_now,
                         const double c);

  std::vector<double> get_c_set_right_opt(const Eigen::VectorXd &w_star,
                                          const double &c_now);
  std::vector<double> get_c_set_right_subopt(const Eigen::VectorXd &w,
                                             const Eigen::VectorXd &grad_w,
                                             const double &c_now);
  std::vector<double> get_c_set_left_subopt(const Eigen::VectorXd &w,
                                            const Eigen::VectorXd &grad_w,
                                            const double &c_now);

  std::vector<double> get_c_set_right_opt_for_path(const Eigen::VectorXd &w,
                                                   const double &c_now);
  std::vector<double>
  get_c_set_right_subopt_for_path(const Eigen::VectorXd &w,
                                  const Eigen::VectorXd &grad_w,
                                  const double &c_now, int &num_dif_ublb);

private:
  Function *fun_obj;
  const double eps;
  const int max_iter;

  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x_;
  Eigen::ArrayXd valid_y_;
  int valid_l_;
  int valid_n_;

  Eigen::VectorXd xm_;
  Eigen::VectorXd valid_x_norm_;
  Eigen::VectorXd w_;
  Eigen::VectorXd grad_;
  double grad_norm_;
};

#endif
