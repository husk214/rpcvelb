#ifndef CTGP_H_
#define CTGP_H_

// for checking theoretical garantee (regularized hyper-)parameter (CTGP)

#include <chrono>
#include <map>
#include <queue>
#include <Eigen/LU>
#include <Eigen/Cholesky>

#include "Dual_solver.hpp"
#include "Primal_solver.hpp"
#include "Stats.hpp"

namespace sdm {

class Ctgp {
public:
  Ctgp(Primal_solver *sol_obj, const double &inexact_level = 1e-3,
       const double c_min = 1e-3, const double c_max = 1e+3,
       const double min_move_c = 1e-6);

  Ctgp(std::vector<Primal_solver *> psol_objs,
       const double &inexact_level = 1e-3, const double c_min = 1e-3,
       const double c_max = 1e+3, const double min_move_c = 1e-6);

  Ctgp(Dual_solver *sol_obj, const double &inexact_level = 1e-3,
       const double c_min = 1e-3, const double c_max = 1e+3,
       const double min_move_c = 1e-6);
  ~Ctgp();

  Primal_solver *psol_obj_;
  Dual_solver *dsol_obj_;
  double inexact_level_;
  double c_min_;
  double c_max_;
  double min_move_c_;

  int train_l_;
  int valid_l_;

  int best_miss_;
  double best_valid_err_;
  double worst_lb_;
  double worst_lb_c_;
  double epsilon_;

  // for cross validation
  std::vector<Primal_solver *> psol_objs_;
  int fold_num_;
  int whole_l_;

  void update_best(const int &num_ub);
  void update_best(const double &valid_err);
  void update_worst_lbve(const double &lbve);
  void update_worst_lbve(const double &lbve, const double &c_key);
  void update_epsilon(const double &sub_worst_lb);

  double find_worst_lb(const double &c1, const double &c2,
                           const Eigen::VectorXd &w_c2,
                           const Eigen::VectorXd &grad_c2) const;
  double find_worst_lb(const double &c1, const Eigen::VectorXd &w_c1,
                           const Eigen::VectorXd &grad_c1, const double &c2,
                           const Eigen::VectorXd &w_c2,
                           const Eigen::VectorXd &grad_c2) const;

  double find_worst_lb(const double &c1, const double &c2,
                           const std::vector<double> &c2_lb) const;
  double find_worst_lb(const double &c1,
                           const std::vector<double> &c1_lb_right,
                           const double &c2,
                           const std::vector<double> &c2_lb_left) const;
  double find_worst_lb(const double &c1,
                           const std::vector<double> &c1_lb_right,
                           const double &c2,
                           const std::vector<double> &c2_lb_left,
                           double &midpoint) const;

};

} // namespace sdm
#endif
