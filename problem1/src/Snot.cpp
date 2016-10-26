#include "Snot.h"

namespace sdm {

Snot::Snot(Primal_solver *sol_obj, const double &rigor, const double c_min,
           const double c_max, const double min_move_c)
    : Ctgp(sol_obj, 1e-3, c_min, c_max, min_move_c) {
  rigor_ = rigor;
}

Snot::Snot(std::vector<Primal_solver *> psol_objs, const double &rigor,
           const double c_min, const double c_max, const double min_move_c)
    : Ctgp(psol_objs, 1e-3, c_min, c_max, min_move_c) {
  rigor_ = rigor;
}

Snot::~Snot() {}

//////////////////////////////
// begin for with grid search
//////////////////////////////

double Snot::with_grid_primal(const int &num_grid) {
  int num_ub, num_lb;
  int train_n = psol_obj_->get_train_n();

  double grid_interval = log10(c_max_ / c_min_) / num_grid;
  double c_now = pow(10, log10(c_min_) + grid_interval);

  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(train_n);
  psol_obj_->set_regularized_parameter(c_now);
  w_now = psol_obj_->train_warm_start_inexact(w_now, epsilon_ * rigor_, num_ub,
                                              num_lb);
  double tmp_worst_lb =
      find_worst_lb(c_min_, c_now, w_now, psol_obj_->get_grad());
  update_worst_lbve(tmp_worst_lb);
  update_best(num_ub);
  double c_pre = c_now;
  Eigen::VectorXd w_pre = w_now;
  Eigen::VectorXd grad_pre = psol_obj_->get_grad();

  for (int iter = 2; iter < num_grid; ++iter) {
    c_now = pow(10, log10(c_min_) + iter * grid_interval);
    psol_obj_->set_regularized_parameter(c_now);
    w_now = psol_obj_->train_warm_start_inexact(w_pre, epsilon_ * rigor_,
                                                num_ub, num_lb);
    tmp_worst_lb = find_worst_lb(c_pre, w_pre, grad_pre, c_now, w_now,
                                 psol_obj_->get_grad());
    update_worst_lbve(tmp_worst_lb);
    update_best(num_ub);
    c_pre = c_now;
    w_pre.noalias() = w_now;
    grad_pre.noalias() = psol_obj_->get_grad();
  }

  c_now = pow(10, log10(c_min_) + num_grid * grid_interval);
  psol_obj_->set_regularized_parameter(c_now);
  w_now = psol_obj_->train_warm_start_inexact(w_pre, epsilon_ * rigor_, num_ub,
                                              num_lb);
  tmp_worst_lb = find_worst_lb(c_max_, c_now, w_now, psol_obj_->get_grad());
  update_worst_lbve(tmp_worst_lb);
  update_best(num_ub);
  // std::cout << "best ve : " << static_cast<double>(best_miss_) / valid_l_
  //           << ", worst_lb : " << worst_lb_ <<", eps : " <<
  //           static_cast<double>(best_miss_) / valid_l_ - worst_lb_ <<
  //           std::endl;
  std::cout << num_grid << " " << epsilon_ << std::endl;
  return (static_cast<double>(best_miss_) / valid_l_) - worst_lb_;
}

////////////////////////////////////////
// begin for with bayesian opti
////////////////////////////////////////

double Snot::with_bayesian_opti_primal(const int &num_trials,
                                       const int &num_init, const double &xi) {
  // int trials = 2;
  initial_gaussian_process_prior(num_init);
  int train_n = psol_obj_->get_train_n();
  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(train_n);
  int num_ub, num_lb;
  double tmp_worst_lb1, tmp_worst_lb2, c1, c2, c_now;
  std::vector<double> c_now_lb_right, c_now_lb_left, c1_lb_right, c2_lb_left;
  Eigen::VectorXd w_c1;
  for (int i = num_init; i < num_trials; ++i) {
    c_now = get_argmax_af(1000, xi);
    auto it = c_interval_map_.upper_bound(c_now);
    --it;
    if (static_cast<int>(c_interval_map_.size()) == 1)
      it = c_interval_map_.begin();
    w_c1 = it->second.w_c1;
    psol_obj_->set_regularized_parameter(c_now);
    w_now = psol_obj_->train_warm_start_inexact(w_now, epsilon_ * rigor_,
                                                num_ub, num_lb);
    update_best(num_ub);
    update_gaussian_process_prior(num_ub, c_now);
    c1 = it->second.c1;
    c2 = it->second.c2;
    c1_lb_right = it->second.c1_lb_right;
    c2_lb_left = it->second.c2_lb_left;

    c_now_lb_left = psol_obj_->get_c_set_left_subopt();
    tmp_worst_lb1 = find_worst_lb(c1, c1_lb_right, c_now, c_now_lb_left);
    c_now_lb_right = psol_obj_->get_c_set_right_subopt();
    tmp_worst_lb2 = find_worst_lb(c_now, c_now_lb_right, c2, c2_lb_left);

    auto range = control_worst_lb_.equal_range(it->second.worst_lb_interval);
    for (auto itera = range.first; itera != range.second; ++itera) {
      if ((itera->second) == c1) {
        control_worst_lb_.erase(itera);
        control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb1, c1));
        control_worst_lb_.insert(
            std::pair<double, double>(tmp_worst_lb2, c_now));
        break;
      }
    }

    c_interval_map_.erase(it);
    c_interval_map_.insert(std::pair<double, C_interval>(
        c1, {c1, c_now, tmp_worst_lb1, c1_lb_right, c_now_lb_left, w_now}));
    c_interval_map_.insert(std::pair<double, C_interval>(
        c_now, {c_now, c2, tmp_worst_lb2, c_now_lb_right, c2_lb_left, w_now}));
    std::cout << i + 1 << " " << c_now << " "
              << static_cast<double>(num_ub) / valid_l_ << " " << epsilon_
              << " " << control_worst_lb_.begin()->second << std::endl;
  }

  return 0.0;
}

double Snot::convert_logit(const double &pos) const {
  return log(pos / (1 - pos));
}

double Snot::convert_inverse_logit(const double &pos) const {
  return exp(pos) / (exp(pos) + 1);
}

void Snot::initial_gaussian_process_prior(const int &num_init) {
  int train_n = psol_obj_->get_train_n();
  int num_ub = 0, num_lb = 0, miss = 0;
  double grid_interval = log10(c_max_ / c_min_) / (num_init - 1.0);
  double c_init, c_pre = c_min_;
  double tmp_worst_lb = 0.0;
  std::vector<double> c_lb_left, c_lb_right_pre;
  Eigen::VectorXd w_now = Eigen::VectorXd(train_n);
  f1_n_.resize(num_init);

  // 1st train in c_min_
  psol_obj_->set_regularized_parameter(c_min_);
  w_now = psol_obj_->train_warm_start_inexact(w_now, epsilon_ * rigor_, num_ub,
                                              num_lb);
  update_best(num_ub);
  f1_n_[0] = -static_cast<double>(num_ub) / valid_l_;
  trained_c_.push_back(log10(c_min_));
  c_lb_right_pre = psol_obj_->get_c_set_right_subopt(
      c_min_, w_now, psol_obj_->get_grad(), num_lb);
  std::cout << "1 " << c_min_ << " " << -1.0 * f1_n_[0] << " "
            << -1.0 * f1_n_[0] << std::endl;

  for (int i = 1; i < num_init - 1; ++i) {
    c_init = pow(10, log10(c_min_) + i * grid_interval);
    psol_obj_->set_regularized_parameter(c_init);
    w_now = psol_obj_->train_warm_start_inexact(w_now, epsilon_ * rigor_,
                                                num_ub, num_lb);
    update_best(num_ub);
    f1_n_[i] = -static_cast<double>(num_ub) / valid_l_;
    trained_c_.push_back(log10(c_init));
    c_lb_left = psol_obj_->get_c_set_left_subopt(c_init, w_now,
                                                 psol_obj_->get_grad(), miss);

    tmp_worst_lb = find_worst_lb(c_pre, c_lb_right_pre, c_init, c_lb_left);
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c_pre));
    c_interval_map_.insert(std::pair<double, C_interval>(
        c_pre,
        {c_pre, c_init, tmp_worst_lb, c_lb_right_pre, c_lb_left, w_now}));

    std::cout << i + 1 << " " << c_init << " " << -1.0 * f1_n_[i] << " "
              << epsilon_ << std::endl;
    c_pre = c_init;
    c_lb_right_pre = psol_obj_->get_c_set_right_subopt(
        c_init, w_now, psol_obj_->get_grad(), miss);
  }

  // last training in c_max_
  c_init = pow(10, log10(c_min_) + (num_init - 1.0) * grid_interval);

  psol_obj_->set_regularized_parameter(c_init);
  w_now = psol_obj_->train_warm_start_inexact(w_now, epsilon_ * rigor_, num_ub,
                                              num_lb);
  update_best(num_ub);
  f1_n_[num_init - 1] = -static_cast<double>(num_ub) / valid_l_;
  trained_c_.push_back(log10(c_init));
  c_lb_left = psol_obj_->get_c_set_left_subopt(c_init, w_now,
                                               psol_obj_->get_grad(), miss);

  tmp_worst_lb = find_worst_lb(c_pre, c_lb_right_pre, c_init, c_lb_left);
  control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c_pre));
  c_interval_map_.insert(std::pair<double, C_interval>(
      c_pre, {c_pre, c_init, tmp_worst_lb, c_lb_right_pre, c_lb_left, w_now}));

  std::cout << num_init << " " << c_init << " " << -1.0 * f1_n_[num_init - 1]
            << " " << epsilon_ << std::endl;

  kernel_.resize(num_init, num_init);
  double tmp_gk = 0.0;
  for (int i = 0; i < num_init; ++i) {
    kernel_.coeffRef(i, i) = 1.0;
    for (int j = i + 1; j < num_init; ++j) {
      tmp_gk = trained_c_[i] - trained_c_[j];
      tmp_gk = exp(-tmp_gk * tmp_gk * 0.5);
      kernel_.coeffRef(i, j) = tmp_gk;
      kernel_.coeffRef(j, i) = tmp_gk;
    }
  }
  kernel_ldlt_.compute(kernel_);
}

void Snot::update_gaussian_process_prior(const int &num_ub,
                                         const double &c_tmp) {
  double c_tmp_log10 = log10(c_tmp);
  trained_c_.push_back(c_tmp_log10);
  int kernel_size = static_cast<int>(kernel_.rows());
  int new_size = kernel_size + 1;
  double tmp_gk = 0.0;
  kernel_.conservativeResize(new_size, new_size);
  kernel_.coeffRef(kernel_size, kernel_size) = 1.0;
  for (int i = 0; i < new_size; ++i) {
    tmp_gk = trained_c_[i] - c_tmp_log10;
    tmp_gk = exp(-tmp_gk * tmp_gk * 0.5);
    kernel_.coeffRef(kernel_size, i) = tmp_gk;
    kernel_.coeffRef(i, kernel_size) = tmp_gk;
  }
  kernel_ldlt_.compute(kernel_);
  f1_n_.conservativeResize(new_size);
  f1_n_[kernel_size] = -(static_cast<double>(num_ub) / valid_l_);
}

double Snot::get_acquisition_func_ei(const double &c_tmp,
                                     const double &xi) const {
  double c_tmp_log10 = log10(c_tmp);
  int kernel_size = kernel_.rows();
  Eigen::VectorXd small_k(kernel_size);
  for (int i = 0; i != kernel_size; ++i)
    small_k[i] = exp(-(trained_c_[i] - c_tmp_log10) *
                     (trained_c_[i] - c_tmp_log10) / 2.0);
  Eigen::VectorXd k_kernel_inv = kernel_ldlt_.solve(small_k);
  double sigma = 1.0 - k_kernel_inv.dot(small_k);
  double ei = 0.0;
  if (sigma > 0.0) {
    double mu = k_kernel_inv.dot(f1_n_);
    double curr_best_ubve = -(static_cast<double>(best_miss_) / valid_l_);
    double ei_z = (mu - curr_best_ubve - xi) / sigma;
    double small_phi_z = exp(-ei_z * ei_z * 0.5) / std::sqrt(2.0 * M_PI);
    double large_phi_z = norm_cdf(ei_z);
    ei = (mu - curr_best_ubve) * large_phi_z + sigma * small_phi_z;
  }
  // std::cout << c_tmp << ", ei " << ei <<" " << pow(10, ei) << std::endl;
  return ei;
}

double Snot::get_argmax_af(const int &grid_num1, const double &xi) const {
  double grid_interval = log10(c_max_ / c_min_) / grid_num1;
  double c_now = 0.0;
  double max_af = -1e+32, tmp_af = 0.0;
  double argmax_c = 0.0, argmax_c_left = 0.0, argmax_c_right = 0.0;
  for (int i = 0; i != grid_num1 + 1; ++i) {
    c_now = pow(10, log10(c_min_) + i * grid_interval);
    tmp_af = get_acquisition_func_ei(c_now, xi);
    if (max_af < tmp_af) {
      max_af = tmp_af;
      argmax_c = c_now;
      if (i == 0) {
        argmax_c_left = c_min_;
      } else {
        argmax_c_left = pow(10, log10(c_min_) + (i - 1) * grid_interval);
      }
      if (i == grid_num1) {
        argmax_c_right = c_max_;
      } else {
        argmax_c_right = pow(10, log10(c_min_) + (i + 1) * grid_interval);
      }
    }
  }

  max_af = 0.0;
  double grid_interval2 = log10(argmax_c_right / argmax_c_left) / 100;
  for (int i = 0; i < 100; ++i) {
    c_now = pow(10, log10(argmax_c_left) + i * grid_interval2);
    tmp_af = get_acquisition_func_ei(c_now, xi);
    if (max_af < tmp_af) {
      max_af = tmp_af;
      argmax_c = c_now;
    }
  }
  return argmax_c;
}

////////////////////////////////////////
// begin for with concerned worst lb
////////////////////////////////////////
double Snot::with_worst_lb_primal(const int &num_trials) {
  int train_n = psol_obj_->get_train_n();
  int num_ub, num_lb;
  double c1 = c_min_;
  std::vector<double> c1_lb_right;
  Eigen::VectorXd w_c1 = Eigen::VectorXd::Zero(train_n);
  if (num_trials > 0) {
    psol_obj_->set_regularized_parameter(c1);
    w_c1 = psol_obj_->train_warm_start_inexact(w_c1, epsilon_ * rigor_, num_ub,
                                               num_lb);
    c1_lb_right = psol_obj_->get_c_set_right_subopt();
    update_best(num_ub);
    update_worst_lbve(find_worst_lb(c_max_, c1, c1_lb_right));
    std::cout << "1 " << epsilon_ << std::endl;
  }

  double c2 = c_max_;
  double midpoint = 0.0, tmp_worst_lb = 0.0;
  std::vector<double> c2_lb_left;
  Eigen::VectorXd w_c2;
  if (num_trials > 1) {
    psol_obj_->set_regularized_parameter(c2);
    w_c2 = psol_obj_->train_warm_start_inexact(w_c1, epsilon_ * rigor_, num_ub,
                                               num_lb);
    c2_lb_left = psol_obj_->get_c_set_left_subopt();
    update_best(num_ub);
    tmp_worst_lb = find_worst_lb(c1, c1_lb_right, c2, c2_lb_left, midpoint);
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c1));
    trained_map_.insert(std::pair<double, Trained_container>(
        c1, {c2, c1_lb_right, c2_lb_left, midpoint, w_c1}));
    std::cout << "2 " << epsilon_ << std::endl;
  }

  double current_worst_lb_c_key = c_min_;
  double c_now = c_min_;
  std::vector<double> c_now_lb_left, c_now_lb_right;
  double tmp_worst_lb1, tmp_worst_lb2, midpoint1, midpoint2;
  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(train_n);
  Eigen::VectorXd w_tmp;
  for (int i = 3; i <= num_trials; ++i) {
    current_worst_lb_c_key = control_worst_lb_.begin()->second;
    auto it = trained_map_.find(current_worst_lb_c_key);
    c_now = it->second.midpoint;
    psol_obj_->set_regularized_parameter(c_now);
    w_tmp = it->second.w;
    w_now = psol_obj_->train_warm_start_inexact(w_tmp, epsilon_ * rigor_,
                                                num_ub, num_lb);
    update_best(num_ub);
    c_now_lb_left = psol_obj_->get_c_set_left_subopt();
    c1_lb_right = it->second.c_lb_right;
    tmp_worst_lb1 = find_worst_lb(current_worst_lb_c_key, c1_lb_right, c_now,
                                  c_now_lb_left, midpoint1);

    c_now_lb_right = psol_obj_->get_c_set_right_subopt();
    tmp_worst_lb2 = find_worst_lb(c_now, c_now_lb_right, it->second.c_next,
                                  it->second.c_next_lb_left, midpoint2);

    control_worst_lb_.erase(control_worst_lb_.begin());
    control_worst_lb_.insert(
        std::pair<double, double>(tmp_worst_lb1, current_worst_lb_c_key));
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb2, c_now));

    trained_map_.insert(std::pair<double, Trained_container>(
        c_now, {it->second.c_next, c_now_lb_right, it->second.c_next_lb_left,
                midpoint2, w_now}));
    trained_map_.erase(it);
    trained_map_.insert(std::pair<double, Trained_container>(
        current_worst_lb_c_key,
        {c_now, c1_lb_right, c_now_lb_left, midpoint1, w_tmp}));
    std::cout << i << " " << epsilon_ << " " << c_now << std::endl;
  }
  // std::cout <<  static_cast<double>(best_miss_)/valid_l_ <<" " <<
  // control_worst_lb_.begin()->first  <<std::endl;
  return static_cast<double>(best_miss_) / valid_l_ -
         (control_worst_lb_.begin()->first);
}

////////////////////////////////////////
// begin for with bisec
////////////////////////////////////////

double Snot::with_bisec_primal(const int num_trials) {
  std::queue<C_interval> c_intervals;
  int train_n = psol_obj_->get_train_n();
  int num_ub, num_lb;
  double c1 = c_min_;
  std::vector<double> c1_lb_right;
  Eigen::VectorXd w_c1 = Eigen::VectorXd::Zero(train_n);
  if (num_trials > 0) {
    psol_obj_->set_regularized_parameter(c1);
    w_c1 = psol_obj_->train_warm_start_inexact(w_c1, epsilon_ * rigor_, num_ub,
                                               num_lb);
    c1_lb_right = psol_obj_->get_c_set_right_subopt();
    update_best(num_ub);
    update_worst_lbve(find_worst_lb(c_max_, c1, c1_lb_right));
    std::cout << "1 " << epsilon_ << std::endl;
  }

  double c2 = c_max_;
  double tmp_worst_lb = 0.0;
  std::vector<double> c2_lb_left;
  Eigen::VectorXd w_c2;
  if (num_trials > 1) {
    psol_obj_->set_regularized_parameter(c2);
    w_c2 = psol_obj_->train_warm_start_inexact(w_c1, epsilon_ * rigor_, num_ub,
                                               num_lb);
    c2_lb_left = psol_obj_->get_c_set_left_subopt();
    update_best(num_ub);
    tmp_worst_lb = find_worst_lb(c1, c1_lb_right, c2, c2_lb_left);
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c1));
    c_intervals.push({c1, c2, tmp_worst_lb, c1_lb_right, c2_lb_left, w_c1});
    std::cout << "2 " << epsilon_ << std::endl;
  }
  // double current_worst_lb_c_key = c_min_;
  double c_now = c_min_, c_left, c_right;
  std::vector<double> c_now_lb_left, c_now_lb_right;
  double tmp_worst_lb1, tmp_worst_lb2;
  // double midpoint1, midpoint2;
  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(train_n);
  Eigen::VectorXd w_tmp;
  C_interval c_interval_now;
  for (int i = 3; i <= num_trials; ++i) {
    c_interval_now = c_intervals.front();
    c_left = c_interval_now.c1;
    c_right = c_interval_now.c2;
    c_now = pow(10, (log10(c_left) + log10(c_right)) * 0.5);

    psol_obj_->set_regularized_parameter(c_now);
    w_tmp = c_interval_now.w_c1;
    w_now = psol_obj_->train_warm_start_inexact(w_tmp, epsilon_ * rigor_,
                                                num_ub, num_lb);
    update_best(num_ub);
    c_now_lb_left = psol_obj_->get_c_set_left_subopt();

    tmp_worst_lb1 =
        find_worst_lb(c_left, c_interval_now.c1_lb_right, c_now, c_now_lb_left);

    c_now_lb_right = psol_obj_->get_c_set_right_subopt();
    tmp_worst_lb2 = find_worst_lb(c_now, c_now_lb_right, c_right,
                                  c_interval_now.c2_lb_left);

    auto range =
        control_worst_lb_.equal_range(c_interval_now.worst_lb_interval);
    for (auto it = range.first; it != range.second; ++it) {
      if ((it->second) == c_left) {
        control_worst_lb_.erase(it);
        control_worst_lb_.insert(
            std::pair<double, double>(tmp_worst_lb1, c_left));
        control_worst_lb_.insert(
            std::pair<double, double>(tmp_worst_lb2, c_now));
        break;
      }
    }
    c_intervals.push({c_left, c_now, tmp_worst_lb1, c_interval_now.c1_lb_right,
                      c_now_lb_left, w_tmp});
    c_intervals.push({c_now, c_right, tmp_worst_lb2, c_now_lb_right,
                      c_interval_now.c2_lb_left, w_now});
    c_intervals.pop();
    std::cout << i << " " << epsilon_ << std::endl;
  }
  // std::cout <<  static_cast<double>(best_miss_)/valid_l_ <<" " <<
  // control_worst_lb_.begin()->first  <<std::endl;
  return static_cast<double>(best_miss_) / valid_l_ -
         (control_worst_lb_.begin()->first);
}

////////////////////////////////////////
// begin for cross validation
////////////////////////////////////////
double Snot::with_cv_grid_primal(const int &num_grids, const double &specified_eps) {

  start_time_ = std::chrono::steady_clock::now();
  int num_ub, num_lb, total_num_ub = 0;
  int train_n = psol_objs_[0]->get_train_n();

  double grid_interval = log10(c_max_ / c_min_) / (num_grids - 1.0);
  double c_now = c_min_;
  double midpoint;
  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(train_n);
  std::vector<Eigen::VectorXd> ws_now;
  std::vector<double> c_now_lb_right, c_now_lb_left, tmp_c_lb;
  for (int i = 0; i < fold_num_; ++i) {
    psol_objs_[i]->set_regularized_parameter(c_now);
    w_now = psol_objs_[i]->train_warm_start_inexact(w_now, min_move_c_, num_ub,
                                                    num_lb);
    total_num_ub += num_ub;
    ws_now.push_back(w_now);
    tmp_c_lb.clear();
    tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
    c_now_lb_right.insert(c_now_lb_right.end(), tmp_c_lb.begin(),
                          tmp_c_lb.end());
    tmp_c_lb.clear();
    tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
    c_now_lb_left.insert(c_now_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
  }
  update_best(total_num_ub);
  std::sort(c_now_lb_right.begin(), c_now_lb_right.end());
  std::sort(c_now_lb_left.begin(), c_now_lb_left.end(), std::greater<double>());

  double tmp_worst_lb;
  double pre_epsilon = min_move_c_;
  update_worst_lbve(1.0);
  double c_pre = c_now;
  std::vector<Eigen::VectorXd> ws_new;
  std::vector<double> c_new_lb_right;
  for (int iter = 1; iter < num_grids - 1; ++iter) {
    c_now = pow(10, log10(c_min_) + iter * grid_interval);
    total_num_ub = 0;
    c_new_lb_right.clear();
    c_now_lb_left.clear();
    ws_new.clear();
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c_now);
      w_now = psol_objs_[i]->train_warm_start_inexact(ws_now[i], pre_epsilon,
                                                      num_ub, num_lb);
      total_num_ub += num_ub;
      ws_new.push_back(w_now);
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
      c_new_lb_right.insert(c_new_lb_right.end(), tmp_c_lb.begin(),
                            tmp_c_lb.end());
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
      c_now_lb_left.insert(c_now_lb_left.end(), tmp_c_lb.begin(),
                           tmp_c_lb.end());
    }
    update_best(total_num_ub);
    std::sort(c_new_lb_right.begin(), c_new_lb_right.end());
    std::sort(c_now_lb_left.begin(), c_now_lb_left.end(),
              std::greater<double>());

    tmp_worst_lb =
        find_worst_lb(c_pre, c_now_lb_right, c_now, c_now_lb_left, midpoint);
    update_worst_lbve(tmp_worst_lb);
    pre_epsilon = best_valid_err_ - tmp_worst_lb;
    c_pre = c_now;
    ws_new.swap(ws_now);
    c_now_lb_right.swap(c_new_lb_right);
  }

  if (num_grids > 1) {
    c_now = pow(10, log10(c_min_) + (num_grids - 1.0) * grid_interval);
    total_num_ub = 0;
    c_now_lb_left.clear();
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c_now);
      w_now = psol_objs_[i]->train_warm_start_inexact(ws_now[i], pre_epsilon,
                                                      num_ub, num_lb);
      total_num_ub += num_ub;
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
      c_now_lb_left.insert(c_now_lb_left.end(), tmp_c_lb.begin(),
                           tmp_c_lb.end());
    }
    update_best(total_num_ub);
    std::sort(c_now_lb_left.begin(), c_now_lb_left.end(),
              std::greater<double>());
    tmp_worst_lb =
        find_worst_lb(c_pre, c_now_lb_right, c_now, c_now_lb_left, midpoint);
    update_worst_lbve(tmp_worst_lb);
    end_time_ = std::chrono::steady_clock::now();
    std::cout << num_grids << " " << epsilon_ << " " << worst_lb_ << " "
              << best_miss_ << " "
              << 1e-3 *
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time_ - start_time_).count() << std::endl;
  } else {
    end_time_ = std::chrono::steady_clock::now();
    std::cout << num_grids << " " << best_valid_err_ << " " << worst_lb_ << " "
              << best_miss_ << " "
              << 1e-3 *
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time_ - start_time_).count() << std::endl;
  }

  return (static_cast<double>(best_miss_) / whole_l_) - worst_lb_;
}

double Snot::with_cv_worst_lb_primal(const int &num_trials, const double &specified_eps) {
  start_time_ = std::chrono::steady_clock::now();

  int train_n = psol_objs_[0]->get_train_n();
  int num_ub, num_lb, total_num_ub = 0;
  double c1 = c_min_;
  double stopping_criterion = specified_eps;
  if (specified_eps < 0.0)
    stopping_criterion = min_move_c_;
  std::vector<double> c1_lb_right, tmp_c_lb;
  Eigen::VectorXd w_c1 = Eigen::VectorXd::Zero(train_n);
  std::vector<Eigen::VectorXd> ws_c1;
  if (num_trials > 0) {
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c1);
      w_c1 = psol_objs_[i]->train_warm_start_inexact(w_c1, stopping_criterion, num_ub,
                                                     num_lb);
      total_num_ub += num_ub;
      ws_c1.push_back(w_c1);
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
      c1_lb_right.insert(c1_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    std::sort(c1_lb_right.begin(), c1_lb_right.end());
    update_best(total_num_ub);
    update_worst_lbve(find_worst_lb(c_max_, c1, c1_lb_right));
    end_time_ = std::chrono::steady_clock::now();
    if (specified_eps < 0.0)
      std::cout << "1 " << epsilon_ << " " << c1 << " " << best_miss_ << " "
                << 1e-3 *
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_time_ - start_time_).count() << std::endl;
  }

  double c2 = c_max_;
  double midpoint = 0.0, tmp_worst_lb = 0.0;
  std::vector<double> c2_lb_left;
  Eigen::VectorXd w_c2 = w_c1;
  if (num_trials > 1) {
    total_num_ub = 0;
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c2);
      w_c2 = psol_objs_[i]->train_warm_start_inexact(w_c2, stopping_criterion, num_ub,
                                                     num_lb);
      total_num_ub += num_ub;
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
      c2_lb_left.insert(c2_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    std::sort(c2_lb_left.begin(), c2_lb_left.end(), std::greater<double>());
    update_best(total_num_ub);
    tmp_worst_lb = find_worst_lb(c1, c1_lb_right, c2, c2_lb_left, midpoint);
    update_epsilon(tmp_worst_lb);
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c1));
    trained_map_cv_.insert(std::pair<double, Trained_container_cv>(
        c1, {c2, c1_lb_right, c2_lb_left, midpoint, ws_c1}));
    end_time_ = std::chrono::steady_clock::now();
    if (specified_eps < 0.0)
      std::cout << "2 " << epsilon_ << " " << c2 << " " << best_miss_ << " "
                << 1e-3 *
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_time_ - start_time_).count() << std::endl;
  }

  double current_worst_lb_c_key = c_min_;
  double c_now = c_min_;
  std::vector<double> c_now_lb_left, c_now_lb_right;
  double tmp_worst_lb1, tmp_worst_lb2, midpoint1, midpoint2;
  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(train_n);
  int iter = 3;
  for (; iter <= num_trials; ++iter) {
    current_worst_lb_c_key = control_worst_lb_.begin()->second;
    auto it = trained_map_cv_.find(current_worst_lb_c_key);
    c_now = it->second.midpoint;
    ws_c1.clear();
    c_now_lb_left.clear();
    c_now_lb_right.clear();
    total_num_ub = 0;
    if (specified_eps < 0.0)
      stopping_criterion = epsilon_ * rigor_;
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c_now);
      w_now = psol_objs_[i]->train_warm_start_inexact(
          it->second.ws[i], stopping_criterion, num_ub, num_lb);
      total_num_ub += num_ub;
      ws_c1.push_back(w_now);
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
      c_now_lb_left.insert(c_now_lb_left.end(), tmp_c_lb.begin(),
                           tmp_c_lb.end());
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
      c_now_lb_right.insert(c_now_lb_right.end(), tmp_c_lb.begin(),
                            tmp_c_lb.end());
    }
    std::sort(c_now_lb_left.begin(), c_now_lb_left.end(),
              std::greater<double>());
    std::sort(c_now_lb_right.begin(), c_now_lb_right.end());

    update_best(total_num_ub);
    c1_lb_right = it->second.c_lb_right;
    tmp_worst_lb1 = find_worst_lb(current_worst_lb_c_key, c1_lb_right, c_now,
                                  c_now_lb_left, midpoint1);

    tmp_worst_lb2 = find_worst_lb(c_now, c_now_lb_right, it->second.c_next,
                                  it->second.c_next_lb_left, midpoint2);

    control_worst_lb_.erase(control_worst_lb_.begin());
    control_worst_lb_.insert(
        std::pair<double, double>(tmp_worst_lb1, current_worst_lb_c_key));
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb2, c_now));
    update_epsilon(control_worst_lb_.begin()->first);
    trained_map_cv_.insert(std::pair<double, Trained_container_cv>(
        c_now, {it->second.c_next, c_now_lb_right, it->second.c_next_lb_left,
                midpoint2, ws_c1}));
    trained_map_cv_.erase(it);
    trained_map_cv_.insert(std::pair<double, Trained_container_cv>(
        current_worst_lb_c_key,
        {c_now, c1_lb_right, c_now_lb_left, midpoint1, ws_c1}));
    end_time_ = std::chrono::steady_clock::now();
    if (epsilon_ <= specified_eps)
      break;
    if (specified_eps < 0.0)
      std::cout << iter << " " << epsilon_ << " " << c_now << " " << best_miss_
                << " "
                << 1e-3 *
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_time_ - start_time_).count() << std::endl;
  }
  if (specified_eps >= 0.0) {
    std::cout << iter << " "  <<specified_eps <<" "  << epsilon_ << " 0.0 " << best_miss_ << " "
              << 1e-3 *
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time_ - start_time_).count() << std::endl;
  }
  return static_cast<double>(best_miss_) / whole_l_ -
         (control_worst_lb_.begin()->first);
}

double Snot::with_cv_bisec_primal(const int num_trials, const double &specified_eps) {
  start_time_ = std::chrono::steady_clock::now();

  std::queue<C_interval_cv> c_intervals;
  int train_n = psol_objs_[0]->get_train_n();
  int num_ub, num_lb, total_num_ub = 0;
  double c1 = c_min_;
  double stopping_criterion = specified_eps;
  if (specified_eps < 0.0)
    stopping_criterion = min_move_c_;
  std::vector<double> c1_lb_right, tmp_c_lb;
  Eigen::VectorXd w_c1 = Eigen::VectorXd::Zero(train_n);
  std::vector<Eigen::VectorXd> ws_c1;
  if (num_trials > 0) {
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c1);
      w_c1 = psol_objs_[i]->train_warm_start_inexact(w_c1, stopping_criterion, num_ub,
                                                     num_lb);
      total_num_ub += num_ub;
      ws_c1.push_back(w_c1);
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
      c1_lb_right.insert(c1_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    std::sort(c1_lb_right.begin(), c1_lb_right.end());
    update_best(total_num_ub);
    update_worst_lbve(find_worst_lb(c_max_, c1, c1_lb_right));
    end_time_ = std::chrono::steady_clock::now();
    if (specified_eps < 0.0)
      std::cout << "1 " << epsilon_ << " " << c1 << " " << best_miss_ << " "
                << 1e-3 *
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_time_ - start_time_).count() << std::endl;
  }

  double c2 = c_max_;
  double tmp_worst_lb = 0.0;
  std::vector<double> c2_lb_left;
  Eigen::VectorXd w_c2 = w_c1;
  if (num_trials > 1) {
    total_num_ub = 0;
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c2);
      w_c2 = psol_objs_[i]->train_warm_start_inexact(w_c2, stopping_criterion, num_ub,
                                                     num_lb);
      total_num_ub += num_ub;
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
      c2_lb_left.insert(c2_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    std::sort(c2_lb_left.begin(), c2_lb_left.end(), std::greater<double>());
    update_best(total_num_ub);
    tmp_worst_lb = find_worst_lb(c1, c1_lb_right, c2, c2_lb_left);
    update_epsilon(tmp_worst_lb);
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c1));
    c_intervals.push({c1, c2, tmp_worst_lb, c1_lb_right, c2_lb_left, ws_c1});
    end_time_ = std::chrono::steady_clock::now();
    if (specified_eps < 0.0)
      std::cout << "2 " << epsilon_ << " " << c2 << " " << best_miss_ << " "
                << 1e-3 *
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_time_ - start_time_).count() << std::endl;
  }
  double c_now = c_min_, c_left, c_right;
  std::vector<double> c_now_lb_left, c_now_lb_right;
  double tmp_worst_lb1, tmp_worst_lb2;
  Eigen::VectorXd w_now, w_tmp;
  C_interval_cv c_interval_now;
  int iter = 3;
  for (; iter <= num_trials; ++iter) {
    c_interval_now = c_intervals.front();
    c_left = c_interval_now.c1;
    c_right = c_interval_now.c2;
    c_now = pow(10, (log10(c_left) + log10(c_right)) * 0.5);
    total_num_ub = 0;
    ws_c1.clear();
    c_now_lb_right.clear();
    c_now_lb_left.clear();
    if (specified_eps < 0.0)
      stopping_criterion = epsilon_ * rigor_;
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c_now);
      w_now = psol_objs_[i]->train_warm_start_inexact(
          c_interval_now.ws_c1[i], stopping_criterion, num_ub, num_lb);
      ws_c1.push_back(w_now);
      total_num_ub += num_ub;
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
      c_now_lb_right.insert(c_now_lb_right.end(), tmp_c_lb.begin(),
                            tmp_c_lb.end());
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
      c_now_lb_left.insert(c_now_lb_left.end(), tmp_c_lb.begin(),
                           tmp_c_lb.end());
    }
    std::sort(c_now_lb_right.begin(), c_now_lb_right.end());
    std::sort(c_now_lb_left.begin(), c_now_lb_left.end(),
              std::greater<double>());
    update_best(total_num_ub);

    tmp_worst_lb1 =
        find_worst_lb(c_left, c_interval_now.c1_lb_right, c_now, c_now_lb_left);
    tmp_worst_lb2 = find_worst_lb(c_now, c_now_lb_right, c_right,
                                  c_interval_now.c2_lb_left);

    auto range =
        control_worst_lb_.equal_range(c_interval_now.worst_lb_interval);
    for (auto it = range.first; it != range.second; ++it) {
      if ((it->second) == c_left) {
        control_worst_lb_.erase(it);
        control_worst_lb_.insert(
            std::pair<double, double>(tmp_worst_lb1, c_left));
        control_worst_lb_.insert(
            std::pair<double, double>(tmp_worst_lb2, c_now));
        break;
      }
    }
    update_epsilon(control_worst_lb_.begin()->first);
    c_intervals.push({c_left, c_now, tmp_worst_lb1, c_interval_now.c1_lb_right,
                      c_now_lb_left, ws_c1});
    c_intervals.push({c_now, c_right, tmp_worst_lb2, c_now_lb_right,
                      c_interval_now.c2_lb_left, ws_c1});
    c_intervals.pop();
    end_time_ = std::chrono::steady_clock::now();
    if (specified_eps < 0.0) {
      std::cout << iter << " " << epsilon_ << " " << c_now << " " << best_miss_
                << " "
                << 1e-3 *
                       std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_time_ - start_time_).count() << std::endl;
    } else {
      if (epsilon_ <= specified_eps)
        break;
    }
  }
  if (specified_eps >= 0.0) {
    std::cout << iter << " " <<specified_eps <<" " << epsilon_ << " 0.0 " << best_miss_ << " "
              << 1e-3 *
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time_ - start_time_).count() << std::endl;
  }

  return static_cast<double>(best_miss_) / whole_l_ -
         (control_worst_lb_.begin()->first);
}

double Snot::with_cv_bayesian_opti_primal(const int &num_trials,
                                          const int &num_init,
                                          const double &xi) {
  start_time_ = std::chrono::steady_clock::now();
  initial_gaussian_process_prior_cv(num_init);
  int train_n = psol_objs_[0]->get_train_n();
  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(train_n);
  int num_ub, num_lb, total_num_ub = 0;
  double tmp_worst_lb1, tmp_worst_lb2, c1, c2, c_now;
  std::vector<double> c_lb_right, c_lb_left, c1_lb_right, c2_lb_left, tmp_c_lb;
  std::vector<Eigen::VectorXd> ws_now;
  for (int i = num_init; i < num_trials; ++i) {
    c_now = get_argmax_af(1000, xi);
    auto it = c_interval_map_cv_.upper_bound(c_now);
    --it;
    if (static_cast<int>(c_interval_map_cv_.size()) == 1)
      it = c_interval_map_cv_.begin();
    total_num_ub = 0;
    ws_now.clear();
    c_lb_right.clear();
    c_lb_left.clear();
    for (int j = 0; j < fold_num_; ++j) {
      psol_objs_[j]->set_regularized_parameter(c_now);
      w_now = psol_objs_[j]->train_warm_start_inexact(
          it->second.ws_c1[j], epsilon_ * rigor_, num_ub, num_lb);
      total_num_ub += num_ub;
      ws_now.push_back(w_now);
      tmp_c_lb = psol_objs_[j]->get_c_set_right_subopt(false);
      c_lb_right.insert(c_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
      tmp_c_lb = psol_objs_[j]->get_c_set_left_subopt(false);
      c_lb_left.insert(c_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    update_best(total_num_ub);
    update_gaussian_process_prior(total_num_ub, c_now);
    std::sort(c_lb_right.begin(), c_lb_right.end());
    std::sort(c_lb_left.begin(), c_lb_left.end(), std::greater<double>());

    c1 = it->second.c1;
    c2 = it->second.c2;
    c1_lb_right = it->second.c1_lb_right;
    c2_lb_left = it->second.c2_lb_left;

    tmp_worst_lb1 = find_worst_lb(c1, c1_lb_right, c_now, c_lb_left);
    tmp_worst_lb2 = find_worst_lb(c_now, c_lb_right, c2, c2_lb_left);

    auto range = control_worst_lb_.equal_range(it->second.worst_lb_interval);
    for (auto itera = range.first; itera != range.second; ++itera) {
      if ((itera->second) == c1) {
        control_worst_lb_.erase(itera);
        control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb1, c1));
        control_worst_lb_.insert(
            std::pair<double, double>(tmp_worst_lb2, c_now));
        break;
      }
    }
    update_epsilon(control_worst_lb_.begin()->first);
    c_interval_map_cv_.erase(it);
    c_interval_map_cv_.insert(std::pair<double, C_interval_cv>(
        c1, {c1, c_now, tmp_worst_lb1, c1_lb_right, c_lb_left, ws_now}));
    c_interval_map_cv_.insert(std::pair<double, C_interval_cv>(
        c_now, {c_now, c2, tmp_worst_lb2, c_lb_right, c2_lb_left, ws_now}));
    end_time_ = std::chrono::steady_clock::now();
    std::cout << i + 1 << " " << epsilon_ << " " << c_now << " " << best_miss_
              << " "
              << 1e-3 *
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time_ - start_time_).count() << std::endl;
  }

  return 0.0;
}

void Snot::initial_gaussian_process_prior_cv(const int &num_init) {

  int train_n = psol_objs_[0]->get_train_n();
  int num_ub = 0, num_lb = 0, total_num_ub = 0;
  double grid_interval = log10(c_max_ / c_min_) / (num_init - 1.0);
  double c_init, c_pre = c_min_;
  double tmp_worst_lb = 0.0;
  std::vector<double> c_lb_left, c_lb_right, c_lb_right_pre, tmp_c_lb;
  Eigen::VectorXd w_now = Eigen::VectorXd(train_n);
  std::vector<Eigen::VectorXd> ws_now, ws_pre;
  f1_n_.resize(num_init);
  // 1st train in c_min_
  for (int i = 0; i < fold_num_; ++i) {
    psol_objs_[i]->set_regularized_parameter(c_min_);
    w_now = psol_objs_[i]->train_warm_start_inexact(w_now, min_move_c_, num_ub,
                                                    num_lb);
    total_num_ub += num_ub;
    ws_pre.push_back(w_now);
    tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(
        c_min_, w_now, psol_objs_[i]->get_grad(), num_lb);
    c_lb_right_pre.insert(c_lb_right_pre.end(), tmp_c_lb.begin(),
                          tmp_c_lb.end());
  }
  update_best(total_num_ub);
  std::sort(c_lb_right_pre.begin(), c_lb_right_pre.end());

  f1_n_[0] = -static_cast<double>(total_num_ub) / valid_l_;
  trained_c_.push_back(log10(c_min_));
  end_time_ = std::chrono::steady_clock::now();
  std::cout << "1 " << -1.0 * f1_n_[0] << " " << c_min_ << " " << total_num_ub
            << " "
            << 1e-3 *
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time_ - start_time_).count() << std::endl;

  for (int i = 1; i < num_init - 1; ++i) {
    c_init = pow(10, log10(c_min_) + i * grid_interval);
    total_num_ub = 0;
    c_lb_left.clear();
    c_lb_right.clear();
    ws_now.clear();
    for (int j = 0; j < fold_num_; ++j) {
      psol_objs_[j]->set_regularized_parameter(c_init);
      w_now = psol_objs_[j]->train_warm_start_inexact(ws_pre[j], min_move_c_,
                                                      num_ub, num_lb);
      total_num_ub += num_ub;
      ws_now.push_back(w_now);
      tmp_c_lb = psol_objs_[j]->get_c_set_right_subopt(
          c_init, w_now, psol_objs_[j]->get_grad(), num_lb);
      c_lb_right.insert(c_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
      tmp_c_lb = psol_objs_[j]->get_c_set_left_subopt(
          c_init, w_now, psol_objs_[j]->get_grad(), num_lb);
      c_lb_left.insert(c_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    update_best(total_num_ub);
    std::sort(c_lb_right.begin(), c_lb_right.end());
    std::sort(c_lb_left.begin(), c_lb_left.end(), std::greater<double>());

    f1_n_[i] = -static_cast<double>(total_num_ub) / valid_l_;
    trained_c_.push_back(log10(c_init));

    tmp_worst_lb = find_worst_lb(c_pre, c_lb_right_pre, c_init, c_lb_left);
    control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c_pre));
    update_epsilon(control_worst_lb_.begin()->first);
    c_interval_map_cv_.insert(std::pair<double, C_interval_cv>(
        c_pre,
        {c_pre, c_init, tmp_worst_lb, c_lb_right_pre, c_lb_left, ws_now}));
    end_time_ = std::chrono::steady_clock::now();
    std::cout << i + 1 << " " << epsilon_ << " " << c_init << " " << best_miss_
              << " "
              << 1e-3 *
                     std::chrono::duration_cast<std::chrono::milliseconds>(
                         end_time_ - start_time_).count() << std::endl;
    c_pre = c_init;
    c_lb_right_pre.swap(c_lb_right);
    ws_pre.swap(ws_now);
  }

  // last training in c_max_
  c_init = pow(10, log10(c_min_) + (num_init - 1.0) * grid_interval);

  total_num_ub = 0;
  c_lb_left.clear();
  for (int i = 0; i < fold_num_; ++i) {
    psol_objs_[i]->set_regularized_parameter(c_init);
    w_now = psol_objs_[i]->train_warm_start_inexact(ws_pre[i], min_move_c_,
                                                    num_ub, num_lb);
    total_num_ub += num_ub;
    tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(
        c_init, w_now, psol_objs_[i]->get_grad(), num_lb);
    c_lb_left.insert(c_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
  }
  update_best(total_num_ub);
  std::sort(c_lb_left.begin(), c_lb_left.end(), std::greater<double>());
  f1_n_[num_init - 1] = -static_cast<double>(total_num_ub) / valid_l_;
  trained_c_.push_back(log10(c_init));

  tmp_worst_lb = find_worst_lb(c_pre, c_lb_right_pre, c_init, c_lb_left);
  control_worst_lb_.insert(std::pair<double, double>(tmp_worst_lb, c_pre));
  c_interval_map_cv_.insert(std::pair<double, C_interval_cv>(
      c_pre, {c_pre, c_init, tmp_worst_lb, c_lb_right_pre, c_lb_left, ws_pre}));

  end_time_ = std::chrono::steady_clock::now();
  std::cout << num_init << " " << epsilon_ << " " << c_init << " " << best_miss_
            << " "
            << 1e-3 *
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time_ - start_time_).count() << std::endl;

  kernel_.resize(num_init, num_init);
  double tmp_gk = 0.0;
  for (int i = 0; i < num_init; ++i) {
    kernel_.coeffRef(i, i) = 1.0;
    for (int j = i + 1; j < num_init; ++j) {
      tmp_gk = trained_c_[i] - trained_c_[j];
      tmp_gk = exp(-tmp_gk * tmp_gk * 0.5);
      kernel_.coeffRef(i, j) = tmp_gk;
      kernel_.coeffRef(j, i) = tmp_gk;
    }
  }
  kernel_ldlt_.compute(kernel_);
}

} // naespace sdm
