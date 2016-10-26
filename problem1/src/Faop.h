#ifndef FAOP_H_
#define FAOP_H_

// for find approximately optimal parameter

#include "Ctgp.h"

namespace sdm {

struct C_subinterval {
  double c_left;
  double c_right;
};

class Faop : public Ctgp {
public:
  Faop(Dual_solver *sol_obj, const double c_min = 1e-3,
       const double c_max = 1e+3, const double min_move_c = 1e-6);
  Faop(Primal_solver *sol_obj, const double &inexact_level = 1e-3,
       const double c_min = 1e-3, const double c_max = 1e+3,
       const double min_move_c = 1e-6);
  Faop(std::vector<Primal_solver *> psol_objs,
       const double &inexact_level = 1e-3, const double c_min = 1e-3,
       const double c_max = 1e+3, const double min_move_c = 1e-6);
  ~Faop();

  void find_app_opt(const double &epsilon);
  void find_app_subopt_12(const int &num_points, const double &epsilon,
                          const double &alpha);
  void find_cv_app_subopt_12(const int &num_points, const double &epsilon,
                           const double &alpha);


private:
  std::multimap<double, C_subinterval> control_worst_lb_;

  double get_apprx_c_right_inexact(const Eigen::VectorXd &w,
                                   const Eigen::VectorXd &grad_w,
                                   const double &tolerance,
                                   const double &now_c) const;

  double get_apprx_c_right_inexact(const double &now_c,
                                   const std::vector<double> &c_now_lb_right,
                                   const double &tolerance) const;

  double get_apprx_c_left_inexact(const Eigen::VectorXd &w,
                                  const Eigen::VectorXd &grad_w,
                                  const double &tolerance,
                                  const double &now_c) const;

  double get_apprx_c_left_inexact(const double &now_c,
                                  const std::vector<double> &c_now_lb_left,
                                  const double &tolerance) const;

  void recursive_check_inexact(const Eigen::VectorXd &w_c1,
                               const Eigen::VectorXd &w_c2,
                               const Eigen::VectorXd &grad_w1,
                               const Eigen::VectorXd &grad_w2, const double &c1,
                               const double &c2, const double &epsilon,
                               const double &the_intersection, int &iter,
                               int &re_iter);

  void recursive_check_inexact(const double &c1,
                               const std::vector<double> c1_lb_right,
                               const double &c2,
                               const std::vector<double> c2_lb_left,
                               const std::vector<Eigen::VectorXd> &ws,
                               const double &epsilon,
                               const double &the_intersection, int &iter,
                               int &re_iter);
};

} // namespace sdm
#endif
