#ifndef TRON_H_
#define TRON_H_

#include <ctime>
#include <chrono>

#include "Tool_impl.hpp"
#include "Primal_solver.hpp"

namespace sdm {
class Tron : public Primal_solver {
public:
  Tron(Primal_function *fun_obj, const double &stpctr = 1e-6,
       const int &max_iter = 1000);

  Tron(Primal_function *fun_obj, const std::string &valid_libsvm_format,
       const double &stpctr = 1e-6, const int &max_iter = 1000);

  Tron(Primal_function *fun_obj,
       Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x,
       Eigen::ArrayXd valid_y, const double &stpctr = 1e-6,
       const int &max_iter = 1000);

  ~Tron();

  void set_regularized_parameter(const double &c);

  int get_max_iteration();
  double get_stopping_criterion();

  int get_train_n();
  int get_valid_l();
  int get_valid_n();

  Eigen::VectorXd get_w();
  Eigen::VectorXd get_grad();
  double get_grad_norm();

  Eigen::VectorXd tron();
  Eigen::VectorXd tron_cg(const double delta, const Eigen::VectorXd &g,
                          Eigen::VectorXd &r, int &cg_iter, double gnorm);

  Eigen::VectorXd train_warm_start(const Eigen::VectorXd &w_start);
  Eigen::VectorXd train_warm_start_inexact(const Eigen::VectorXd &w,
                                           const double inexact_level,
                                           int &num_ub, int &num_lb);

  double get_valid_error(const Eigen::VectorXd &w);
  void get_ub_lb_ve_apprx(int &num_ub, int &num_lb);

  std::vector<double> get_c_set_right_opt(const double &c_now,
                                          const Eigen::VectorXd &w_star,
                                          int &miss);

  std::vector<double> get_c_set_right_subopt(const bool &flag_sort = true) const;
  std::vector<double> get_c_set_left_subopt(const bool &flag_sort = true) const;

  std::vector<double> get_c_set_right_subopt(const double &c_now,
                                             const Eigen::VectorXd &w,
                                             const Eigen::VectorXd &grad_w,
                                             const bool &flag_sort = true) const;
  std::vector<double> get_c_set_left_subopt(const double &c_now,
                                            const Eigen::VectorXd &w,
                                            const Eigen::VectorXd &grad_w,
                                            const bool &flag_sort = true) const;

  std::vector<double> get_c_set_right_opt_for_path(const Eigen::VectorXd &w,
                                                   const double &c_now);
  std::vector<double>
  get_c_set_right_subopt_for_path(const Eigen::VectorXd &w,
                                  const Eigen::VectorXd &grad_w,
                                  const double &c_now, int &num_dif_ublb);

private:
  Primal_function *fun_obj_;

  double stopping_criterion;
  int max_iteration;

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
}

#endif
