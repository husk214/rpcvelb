#ifndef VALIDATION_ERROR_PATH_H_
#define VALIDATION_ERROR_PATH_H_

#include "Function.hpp"
#include "Solver.hpp"
#include "Tools.h"

#include <algorithm>
#include <iterator>
#include <cfloat>
#include <chrono>

class Validation_error_path {
public:
  Validation_error_path(std::vector<Solver *> train_objs,
                        const double c_min = 1e-6, const double c_max = 1e+3,
                        const double min_move_c = 1e-6);
  Validation_error_path(Solver *train_obj,
                        const std::string fileNameLibSVMFormat,
                        const double c_min = 1e-6, const double c_max = 1e+3,
                        const double min_move_c = 1e-6);
  Validation_error_path(Solver *train_obj,
                        Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x,
                        Eigen::ArrayXd valid_y, const double c_min = 1e-6,
                        const double c_max = 1e+3,
                        const double min_move_c = 1e-6);

  ~Validation_error_path();

  double get_best_c();
  Eigen::VectorXd get_best_w();

  int get_num_opti_call();
  double get_total_time();

  void check_log_scale(const int &num_points, const double &inexact_level);

  void exact_path();
  void exact_best_path();
  void exact_path_only_error();

  void approximate_path(const double &epsilon);
  void approximate_best_path(const double &epsilon);
  void approximate_path_only_error(const double &epsilon);

  void approximate_path_inexact(const double &epsilon,
                                const double &inexact_level);

  void apprx_aggressive(const double &epsilon, const double &aggressive);

  void apprx_inexact_train_multi_aggr(const int &num_points,
                                      const double &epsilon,
                                      const double &inexact_level,
                                      const double &aggressive);

  void approximate_inexact_train(const double &epsilon,
                                 const double &inexact_level);

  void multi_exact(const int &num_points);
  void multi_apprx(const int &num_points, const double &epsilon);

  void multi_exact_inexact_train(const int &num_points);

  void multi_apprx_inexact_train(const int &num_points, const double &epsilon,
                                 const double &inexact_level,
                                 const double &accuracy_binary_search = 1e-6);

  void cross_validation_apprx_exact(const double &epsilon);
  void cross_validation_apprx_exact_multi(const int &num_points,
                                          const double &epsilon);

  void cross_validation_apprx_inexact(const double &epsilon,
                                      const double &inexact_level);

  void cross_validation_apprx_inexact_multi_aggr(const int &num_points,
                                                 const double &epsilon,
                                                 const double &inexact_level,
                                                 const double &aggressive);

  void cross_validation_apprx_exact_path(const double &epsilon);
  void cross_validation_apprx_inexact_path(const double &epsilon,
                                           const double &inexact_level);

  void grid_search_log_scale(const int &nun_grid_points);

protected:
  Solver *train_obj_;
  double c_min_;
  double c_max_;
  double min_move_c_;

  Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x_;
  Eigen::ArrayXd valid_y_;
  int valid_l_;
  int valid_n_;
  Eigen::VectorXd valid_x_norm_;

  // current best informations
  int best_num_error_;
  double best_valid_error_;
  double best_c_;
  Eigen::VectorXd best_w_;

  // for cross validation
  std::vector<Solver *> train_objs_;
  int fold_num_;
  int whole_l_;

  // other informations
  int num_opti_call_;
  double total_time_;

  void clear_best();
  void update_best(const double tmp_valid_err, const double tmp_c);
  void update_best(const int tmp_num_err, const double tmp_valid_err,
                   const double tmp_c);
  void update_best(const int tmp_num_err, const double tmp_valid_err,
                   const double tmp_c, const Eigen::VectorXd &tmp_w);
  double get_valid_error(const Eigen::VectorXd &w);

  double get_c_exact_arc_suboptimal_bound(const Eigen::VectorXd &w,
                                          const double &now_c,
                                          double &valid_err);
  double get_exact_min_c(const Eigen::VectorXd &w, const double &now_c,
                         double &valid_err);
  double get_exact_best_min_c(const Eigen::VectorXd &w, const double &now_c,
                              double &valid_error);

  double get_exact_c_only_error(const Eigen::VectorXd &w, const double &now_c,
                                double &valid_error);

  double get_approximate_min_c(const Eigen::VectorXd &w, const double &now_c,
                               const double &eps, double &valid_error);
  double get_approximate_best_min_c(const Eigen::VectorXd &w,
                                    const double &now_c, const double &eps,
                                    double &valid_error);
  double get_approximate_c_only_error(const Eigen::VectorXd &w,
                                      const double &now_c, const double &eps,
                                      double &valid_error);

  double get_apprx_c_prev(const Eigen::VectorXd &w_star, const double &now_c,
                          const double &eps, double &valid_error);

  double get_apprx_c_bisec_right(const Eigen::VectorXd &w,
                                 const Eigen::VectorXd &grad_w,
                                 const double &tolerance, const double &now_c,
                                 const double &accuracy = 1e-6);

  double
  get_apprx_c_bisec_right_cv(const std::vector<Eigen::VectorXd> &w_set,
                             const std::vector<Eigen::VectorXd> &grad_w_set,
                             const double &tolerance, const double &now_c,
                             const double &accuracy = 1e-6);

  double get_apprx_c_bisec_left(const Eigen::VectorXd &w,
                                const Eigen::VectorXd &grad_w,
                                const double &tolerance, const double &now_c,
                                const double &pre_c,
                                const double &accuracy = 1e-6);
  double
  get_apprx_c_bisec_left_cv(const std::vector<Eigen::VectorXd> &w_set,
                            const std::vector<Eigen::VectorXd> &grad_w_set,
                            const double &tolerance, const double &now_c,
                            const double &pre_c, const double &accuracy = 1e-6);

  double get_apprx_c_right_inexact(const Eigen::VectorXd &w,
                                   const Eigen::VectorXd &grad_w,
                                   const double &tolerance,
                                   const double &now_c);

  double get_apprx_c_right_inexact(const Eigen::VectorXd &w,
                                   const Eigen::VectorXd &grad_w,
                                   const double &tolerance, const double &now_c,
                                   int &c_set_th);

  double get_apprx_c_left_inexact(const Eigen::VectorXd &w,
                                  const Eigen::VectorXd &grad_w,
                                  const double &tolerance, const double &now_c);

  double get_apprx_c_right_exact_cv(const std::vector<Eigen::VectorXd> &w_set,
                                    const double &tolerance,
                                    const double &now_c);

  double
  get_apprx_c_right_inexact_cv(const std::vector<Eigen::VectorXd> &w_set,
                               const std::vector<Eigen::VectorXd> &grad_w_set,
                               const double &tolerance, const double &now_c);

  double
  get_apprx_c_left_inexact_cv(const std::vector<Eigen::VectorXd> &w_set,
                              const std::vector<Eigen::VectorXd> &grad_w_set,
                              const double &tolerance, const double &now_c);

  double
  get_apprx_c_right_exact_cv_for_path(const std::vector<Eigen::VectorXd> &w_set,
                                      const double &tolerance,
                                      const double &now_c);
  double get_apprx_c_right_inexact_cv_for_path(
      const std::vector<Eigen::VectorXd> &w_set,
      const std::vector<Eigen::VectorXd> &grad_w_set, const double &tolerance,
      const double &now_c);

  double get_upper_bound_valid_error(const Eigen::VectorXd &w,
                                     const Eigen::VectorXd &func_grad);
  double get_lower_bound_valid_error(const Eigen::VectorXd &w,
                                     const Eigen::VectorXd &c_loss_grad);

  void recursive_check(const Eigen::VectorXd &w_c1, const Eigen::VectorXd &w_c2,
                       const double &c1, const double &c2,
                       const double &epsilon, int &iter, int &re_iter);
  void recursive_check_bisec(const Eigen::VectorXd &w_c1,
                             const Eigen::VectorXd &w_c2,
                             const Eigen::VectorXd &grad_w1,
                             const Eigen::VectorXd &grad_w2, const double &c1,
                             const double &c2, const double &epsilon,
                             const double &inexact_level, int &iter,
                             int &re_iter);

  void recursive_check_bisec_cv(const std::vector<Eigen::VectorXd> &w_c1,
                                const std::vector<Eigen::VectorXd> &w_c2,
                                const std::vector<Eigen::VectorXd> &grad_w1,
                                const std::vector<Eigen::VectorXd> &grad_w2,
                                const double &c1, const double &c2,
                                const double &epsilon,
                                const double &inexact_level, int &iter,
                                int &re_iter);

  void recursive_check_inexact(const Eigen::VectorXd &w_c1,
                               const Eigen::VectorXd &w_c2,
                               const Eigen::VectorXd &grad_w1,
                               const Eigen::VectorXd &grad_w2, const double &c1,
                               const double &c2, const double &epsilon,
                               const double &inexact_level, int &iter,
                               int &re_iter);
  void recursive_check_inexact_cv(const std::vector<Eigen::VectorXd> &w_c1,
                                  const std::vector<Eigen::VectorXd> &w_c2,
                                  const std::vector<Eigen::VectorXd> &grad_w1,
                                  const std::vector<Eigen::VectorXd> &grad_w2,
                                  const double &c1, const double &c2,
                                  const double &epsilon,
                                  const double &inexact_level, int &iter,
                                  int &re_iter);

  double get_apprx_path_c_inexact(const Eigen::VectorXd &w,
                                  const Eigen::VectorXd &grad_w,
                                  const double &tolerance, const double &now_c);
};

#endif // VALIDATION_ERROR_PATH_H
