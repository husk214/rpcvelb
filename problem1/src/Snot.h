#ifndef SNOT_H_
#define SNOT_H_

// for Ctgp in specified number of times

#include "Ctgp.h"

namespace sdm {

struct C_interval {
  double c1;
  double c2;
  double worst_lb_interval;
  std::vector<double> c1_lb_right;
  std::vector<double> c2_lb_left;
  Eigen::VectorXd w_c1;
};

struct Trained_container {
  double c_next;
  std::vector<double> c_lb_right;
  std::vector<double> c_next_lb_left;
  double midpoint;
  Eigen::VectorXd w;
};

struct C_interval_cv {
  double c1;
  double c2;
  double worst_lb_interval;
  std::vector<double> c1_lb_right;
  std::vector<double> c2_lb_left;
  std::vector<Eigen::VectorXd> ws_c1;
};

struct Trained_container_cv {
  double c_next;
  std::vector<double> c_lb_right;
  std::vector<double> c_next_lb_left;
  double midpoint;
  std::vector<Eigen::VectorXd> ws;
};

class Snot : public Ctgp {
public:
  Snot(Primal_solver *sol_obj, const double &rigor=0.1,
       const double c_min = 1e-3, const double c_max = 1e+3,
       const double min_move_c = 1e-6);
  Snot(std::vector<Primal_solver *> psol_objs,
       const double &rigor=0.1, const double c_min = 1e-3,
       const double c_max = 1e+3, const double min_move_c = 1e-6);
  ~Snot();

  double with_grid_primal(const int &num_grids);
  double with_bayesian_opti_primal(const int &num_trials, const int &num_init,
                                   const double &xi);
  double with_worst_lb_primal(const int &num_trials);
  double with_bisec_primal(const int num_trials);

  // for cross validation
  double with_cv_grid_primal(const int &num_grids, const double &specified_eps = -1.0);
  double with_cv_worst_lb_primal(const int &num_trials, const double &specified_eps = -1.0);
  double with_cv_bisec_primal(const int num_trials, const double &specified_eps = -1.0);

  double with_cv_bayesian_opti_primal(const int &num_trials,
                                      const int &num_init, const double &xi);

private:
  std::chrono::steady_clock::time_point start_time_, end_time_;

  double rigor_;

  // for bayes opt
  Eigen::VectorXd f1_n_;
  Eigen::MatrixXd kernel_;
  Eigen::LDLT<Eigen::MatrixXd> kernel_ldlt_;
  std::multimap<double, Trained_container> trained_map_;
  std::multimap<double, C_interval> c_interval_map_;
  std::multimap<double, double> control_worst_lb_;
  std::vector<double> trained_c_;

  // for cross validation
  std::multimap<double, Trained_container_cv> trained_map_cv_;
  std::multimap<double, C_interval_cv> c_interval_map_cv_;

  // for with baysian optimization primal
  double convert_logit(const double &pos) const;
  double convert_inverse_logit(const double &pos) const;
  void initial_gaussian_process_prior(const int &num_init);
  void initial_gaussian_process_prior_cv(const int &num_init);
  void update_gaussian_process_prior(const int &num_ub, const double &c_tmp);
  double get_acquisition_func_ei(const double &c_tmp, const double &xi) const;
  double get_argmax_af(const int &num_grids, const double &xi) const;
};

} // namespace sdm

#endif
