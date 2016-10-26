#include "Faop.h"

namespace sdm {

Faop::Faop(Primal_solver *sol_obj, const double &inexact_level,
           const double c_min, const double c_max, const double min_move_c)
    : Ctgp(sol_obj, inexact_level, c_min, c_max, min_move_c) {}

Faop::Faop(std::vector<Primal_solver *> psol_objs, const double &inexact_level,
           const double c_min, const double c_max, const double min_move_c)
    : Ctgp(psol_objs, inexact_level, c_min, c_max, min_move_c) {}

Faop::Faop(Dual_solver *sol_obj, const double c_min, const double c_max,
           const double min_move_c)
    : Ctgp(sol_obj, 1e-3, c_min, c_max, min_move_c) {}

Faop::~Faop() {}

void Faop::find_app_opt(const double &epsilon) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  double c_now = c_min_, c_old = c_min_;
  Eigen::ArrayXd alpha = Eigen::ArrayXd::Ones(train_l_);
  dsol_obj_->set_regularized_parameter(c_now);
  unsigned int iter_count = 0, th = 0;
  double valid_err;
  alpha = dsol_obj_->train_warm_start(alpha);
  std::vector<double> c_vec;

  while (1) {
    ++iter_count;
    c_vec = dsol_obj_->get_c_set_right_opt(alpha, c_now, valid_err);
    update_best(valid_err);
    end = std::chrono::system_clock::now();
    diff = end - start;
    th = (valid_err - best_valid_err_ + epsilon) * valid_l_;

    std::cout << iter_count << " " << c_old << " " << valid_err << " "
              << " " << th << " " << c_vec.size() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;
    c_old = c_now;
    if (c_vec.size() > th)
      c_now = c_vec.at(th);
    else
      c_now = c_vec.at(c_vec.size() - 2);

    if (c_now >= c_max_)
      break;
    dsol_obj_->set_regularized_parameter(c_now);
    alpha = dsol_obj_->train_warm_start(alpha);
  }
}

void Faop::find_app_subopt_12(const int &num_points, const double &epsilon,
                              const double &alpha) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = psol_obj_->get_train_n();
  std::vector<Eigen::VectorXd> w_set;
  std::vector<Eigen::VectorXd> grad_set;
  std::vector<double> ub_ve_set;
  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  Eigen::VectorXd w_star = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd grad_w, vw_star, vgrad_w;
  double c_hat, c_hat_old, next_c_hat, tmp_ub_c_hat = 1.0;
  int num_ub, num_lb;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    psol_obj_->set_regularized_parameter(c_hat);
    w_star = psol_obj_->train_warm_start_inexact(w_star, inexact_level_, num_ub,
                                                 num_lb);
    tmp_ub_c_hat = static_cast<double>(num_ub) / valid_l_;
    ub_ve_set.push_back(tmp_ub_c_hat);
    w_set.push_back(w_star);
    grad_set.push_back(psol_obj_->get_grad());
    update_best(num_ub);
  }

  int iter_count = 0, recu_iter = 0;
  Eigen::VectorXd w_tilde, w_tilde_old, grad_w_old;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    c_hat_old = c_hat;
    next_c_hat = pow(10.0, (log_cmin + (i + 1.0) * log_interval));
    w_tilde = w_set[i];
    w_tilde_old = w_tilde;
    grad_w = grad_set[i];
    grad_w_old = grad_w;
    ++iter_count;

    end = std::chrono::system_clock::now();
    diff = end - start;
    // std::cout << iter_count << " " << c_hat << " " << ub_ve_set[i] << " "
    //           // << psol_obj_->get_grad_norm() << " "
    //           << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
    //                   .count() /
    //               1000.0) << " 0.0" << std::endl;

    while (c_hat <= next_c_hat) {
      c_hat = get_apprx_c_right_inexact(w_tilde, grad_w, alpha, c_hat);
      if (c_hat > next_c_hat) {
        c_hat = get_apprx_c_right_inexact(w_tilde, grad_w, epsilon, c_hat);
        if (c_hat > next_c_hat)
          break;
      }
      psol_obj_->set_regularized_parameter(c_hat);

      w_tilde = psol_obj_->train_warm_start_inexact(w_tilde, inexact_level_,
                                                    num_ub, num_lb);
      grad_w = psol_obj_->get_grad();
      update_best(num_ub);
      double tmp_intersection = find_worst_lb(
          c_hat_old, w_tilde_old, grad_w_old, c_hat, w_tilde, grad_w);
      control_worst_lb_.insert(std::pair<double, C_subinterval>(
          tmp_intersection, {c_hat_old, c_hat}));
      update_epsilon(control_worst_lb_.begin()->first);
      if (alpha > epsilon)
        recursive_check_inexact(w_tilde_old, w_tilde, grad_w_old, grad_w,
                                c_hat_old, c_hat, epsilon, tmp_intersection,
                                iter_count, recu_iter);
      ++iter_count;
      end = std::chrono::system_clock::now();
      diff = end - start;
      // std::cout << iter_count << " " << c_hat << " " << ub_c_hat << " "
      //           // << psol_obj_->get_grad_norm() << " "
      //           <<
      //           (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
      //                   .count() /
      //               1000.0) << " " << ub_c_hat - lb_c_hat << std::endl;
      c_hat_old = c_hat;
      w_tilde_old.noalias() = w_tilde;
      grad_w_old.noalias() = grad_w;
    }
    // total_time_ =
    //     (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count()
    //     /
    //      1000.0);
  }
  // std::cout << '\n' << "total iter: " << iter_count
  //           << ", recursive_iter: " << recu_iter
  //           << ", best ve: " << best_valid_err_ //<< ", c:" << best_c_
  //           << ", worst lb : " << (control_worst_lb_.begin())->first
  //           << ", grantee eps : " << best_valid_err_ -
  //           (control_worst_lb_.begin())->first
  //           << ", time: "
  //           << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
  //                   .count() /
  //               1000.0) << std::endl;
  std::cout << iter_count << " " << epsilon_ << std::endl;
}

void Faop::find_cv_app_subopt_12(const int &num_points, const double &epsilon,
                                 const double &alpha) {

  std::chrono::steady_clock::time_point start =
      std::chrono::steady_clock::now();

  int n = psol_objs_[0]->get_train_n();
  std::vector<double> ub_ve_set;
  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  Eigen::VectorXd w_now = Eigen::VectorXd::Zero(n);
  double c_hat, c_hat_old, next_c_hat, tmp_ub_c_hat = 1.0;
  int num_ub, num_lb, total_num_ub = 0;
  std::vector<Eigen::VectorXd> ws_pre, ws_now, ws_zero;
  std::vector<double> c_lb_left, c_lb_right, tmp_c_lb;
  std::vector<std::vector<double>> c_lbs_left, c_lbs_right;
  // train c_min_
  for (int i = 0; i < fold_num_; ++i) {
    psol_objs_[i]->set_regularized_parameter(c_min_);
    w_now = psol_objs_[i]->train_warm_start_inexact(w_now, inexact_level_,
                                                    num_ub, num_lb);
    total_num_ub += num_ub;
    ws_pre.push_back(w_now);
    ws_zero.push_back(w_now);
    tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
    c_lb_right.insert(c_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
  }
  std::sort(c_lb_right.begin(), c_lb_right.end());
  c_lbs_right.push_back(c_lb_right);
  update_best(total_num_ub);

  // train for trick 1
  for (int i = 1; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    total_num_ub = 0;
    ws_now.clear();
    c_lb_right.clear();
    c_lb_left.clear();
    for (int j = 0; j < fold_num_; ++j) {
      psol_objs_[j]->set_regularized_parameter(c_hat);
      w_now = psol_objs_[j]->train_warm_start_inexact(ws_pre[j], inexact_level_,
                                                      num_ub, num_lb);
      total_num_ub += num_ub;
      ws_now.push_back(w_now);
      tmp_c_lb = psol_objs_[j]->get_c_set_right_subopt(false);
      c_lb_right.insert(c_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
      tmp_c_lb = psol_objs_[j]->get_c_set_left_subopt(false);
      c_lb_left.insert(c_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    update_best(total_num_ub);
    std::sort(c_lb_right.begin(), c_lb_right.end());
    std::sort(c_lb_left.begin(), c_lb_left.end(), std::greater<double>());
    c_lbs_right.push_back(c_lb_right);
    c_lbs_left.push_back(c_lb_left);
    ws_now.swap(ws_pre);
    tmp_ub_c_hat = static_cast<double>(total_num_ub) / valid_l_;
    ub_ve_set.push_back(tmp_ub_c_hat);
  }

  int iter_count = 0, recu_iter = 0;
  std::vector<double> c_old_lb_right;
  ws_pre.swap(ws_zero);
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    c_hat_old = c_hat;
    next_c_hat = pow(10.0, (log_cmin + (i + 1.0) * log_interval));
    ++iter_count;

    // std::cout << iter_count << " " << c_hat << " " << ub_ve_set[i] << " "
    //           // << psol_obj_->get_grad_norm() << " "
    //           << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
    //                   .count() /
    //               1000.0) << " 0.0" << std::endl;
    c_old_lb_right.swap(c_lbs_right[i]);
    while (c_hat <= next_c_hat) {
      c_hat = get_apprx_c_right_inexact(c_hat, c_old_lb_right, alpha);
      if (c_hat > next_c_hat) {
        if (i != num_points - 1) {
          double tmp_intersection = find_worst_lb(c_hat_old, c_old_lb_right,
                                                  next_c_hat, c_lbs_left[i]);
          recursive_check_inexact(c_hat_old, c_old_lb_right, next_c_hat,
                                  c_lbs_left[i], ws_pre, epsilon,
                                  tmp_intersection, iter_count, recu_iter);
          break;
        } else {
          c_hat = get_apprx_c_right_inexact(c_hat_old, c_old_lb_right, epsilon);
          if (c_hat > next_c_hat)
            break;
        }
      }
      total_num_ub = 0;
      ws_now.clear();
      c_lb_right.clear();
      c_lb_left.clear();
      for (int j = 0; j < fold_num_; ++j) {
        psol_objs_[j]->set_regularized_parameter(c_hat);
        w_now = psol_objs_[j]->train_warm_start_inexact(
            ws_pre[j], inexact_level_, num_ub, num_lb);
        total_num_ub += num_ub;
        ws_now.push_back(w_now);
        tmp_c_lb = psol_objs_[j]->get_c_set_right_subopt(false);
        c_lb_right.insert(c_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
        tmp_c_lb = psol_objs_[j]->get_c_set_left_subopt(false);
        c_lb_left.insert(c_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
      }
      update_best(total_num_ub);
      std::sort(c_lb_right.begin(), c_lb_right.end());
      std::sort(c_lb_left.begin(), c_lb_left.end(), std::greater<double>());

      double tmp_intersection =
          find_worst_lb(c_hat_old, c_old_lb_right, c_hat, c_lb_left);
      control_worst_lb_.insert(std::pair<double, C_subinterval>(
          tmp_intersection, {c_hat_old, c_hat}));
      update_epsilon(control_worst_lb_.begin()->first);

      if (alpha > epsilon)
        recursive_check_inexact(c_hat_old, c_old_lb_right, c_hat, c_lb_left,
                                ws_pre, epsilon, tmp_intersection, iter_count,
                                recu_iter);
      c_hat_old = c_hat;
      ws_pre.swap(ws_now);
      c_old_lb_right.swap(c_lb_right);
      ++iter_count;

      // std::cout << iter_count << " " << c_hat << " " << ub_c_hat << " "
      //           // << psol_obj_->get_grad_norm() << " "
      //           <<
      //           (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
      //                   .count() /
      //               1000.0) << " " << ub_c_hat - lb_c_hat << std::endl;
    }
  }
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << iter_count << " " << epsilon_ << " 0.0 " << best_miss_ << " "
            << 1e-3 *
                   std::chrono::duration_cast<std::chrono::milliseconds>(
                       end - start).count() << std::endl;
}

double Faop::get_apprx_c_right_inexact(const Eigen::VectorXd &w,
                                       const Eigen::VectorXd &grad_w,
                                       const double &tolerance,
                                       const double &now_c) const {
  int tolerance_instance = valid_l_ * tolerance;
  std::vector<double> c_vec =
      psol_obj_->get_c_set_right_subopt(now_c, w, grad_w);
  double next_c;
  int th = static_cast<int>(c_vec.size()) - best_miss_ + tolerance_instance;

  if (c_vec.empty())
    return c_max_;
  if (th < 1) {
    next_c = c_vec.at(0);
  } else if (th <= c_vec.size() - 1) {
    next_c = c_vec.at(th);
  } else {
    next_c = c_vec.at(c_vec.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Faop::get_apprx_c_left_inexact(const Eigen::VectorXd &w,
                                      const Eigen::VectorXd &grad_w,
                                      const double &tolerance,
                                      const double &now_c) const {
  int tolerance_instance = valid_l_ * tolerance;
  std::vector<double> c_vec =
      psol_obj_->get_c_set_left_subopt(now_c, w, grad_w);

  double next_c;
  int th = static_cast<int>(c_vec.size()) - best_miss_ + tolerance_instance;

  if (c_vec.empty())
    return c_min_;
  if (th < 1) {
    next_c = c_vec.at(0);
  } else if (th <= (static_cast<int>(c_vec.size()) - 1)) {
    next_c = c_vec.at(th);
  } else {
    next_c = c_vec.at(c_vec.size() - 1);
  }

  if (now_c - next_c < min_move_c_)
    next_c = now_c - min_move_c_;

  return next_c;
}

double
Faop::get_apprx_c_right_inexact(const double &now_c,
                                const std::vector<double> &c_now_lb_right,
                                const double &tolerance) const {
  int tolerance_instance = valid_l_ * tolerance;
  int miss = c_now_lb_right.size();
  double next_c;
  int th = miss - best_miss_ + tolerance_instance;

  if (c_now_lb_right.empty())
    return c_max_;
  if (th < 1) {
    next_c = c_now_lb_right.at(0);
  } else if (th <= c_now_lb_right.size() - 1) {
    next_c = c_now_lb_right.at(th);
  } else {
    next_c = c_now_lb_right.at(c_now_lb_right.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Faop::get_apprx_c_left_inexact(const double &now_c,
                                      const std::vector<double> &c_now_lb_left,
                                      const double &tolerance) const {
  int tolerance_instance = valid_l_ * tolerance;
  int miss = c_now_lb_left.size();
  double next_c;
  int th = miss - best_miss_ + tolerance_instance;
  if (c_now_lb_left.empty())
    return c_min_;
  if (th < 1) {
    next_c = c_now_lb_left.at(0);
  } else if (th <= (static_cast<int>(c_now_lb_left.size()) - 1)) {
    next_c = c_now_lb_left.at(th);
  } else {
    next_c = c_now_lb_left.at(c_now_lb_left.size() - 1);
  }

  if (now_c - next_c < min_move_c_)
    next_c = now_c - min_move_c_;

  return next_c;
}

void Faop::recursive_check_inexact(
    const Eigen::VectorXd &w_c1, const Eigen::VectorXd &w_c2,
    const Eigen::VectorXd &grad_w1, const Eigen::VectorXd &grad_w2,
    const double &c1, const double &c2, const double &epsilon,
    const double &the_intersection, int &iter, int &re_iter) {

  if (c2 - c1 <= min_move_c_ || c2 >= c_max_) {
    return;
  }

  double c1_tilde = get_apprx_c_right_inexact(w_c1, grad_w1, epsilon, c1);
  double c2_tilde = get_apprx_c_left_inexact(w_c2, grad_w2, epsilon, c2);
  int num_ub, num_lb;
  Eigen::VectorXd grad_w_cm;
  if (c1_tilde < c2_tilde) {
    double c_m = 0.5 * (c1_tilde + c2_tilde);
    psol_obj_->set_regularized_parameter(c_m);
    Eigen::VectorXd w_cm = psol_obj_->train_warm_start_inexact(
        w_c1, inexact_level_, num_ub, num_lb);

    update_best(num_ub);
    ++re_iter;
    ++iter;
    // std::cout << iter << " " << c_m << " " << num_ub << " " << c1_tilde << "
    // "
    //           << c2_tilde << " " << 0 << std::endl;
    grad_w_cm = psol_obj_->get_grad();
    double tmp_intersection1 =
        find_worst_lb(c1, w_c1, grad_w1, c_m, w_cm, grad_w_cm);
    double tmp_intersection2 =
        find_worst_lb(c_m, w_cm, grad_w_cm, c2, w_c2, grad_w2);

    auto range = control_worst_lb_.equal_range(the_intersection);
    for (auto it = range.first; it != range.second; ++it) {
      if ((it->second).c_left == c1) {
        control_worst_lb_.erase(it);
        control_worst_lb_.insert(
            std::pair<double, C_subinterval>(tmp_intersection1, {c1, c_m}));
        control_worst_lb_.insert(
            std::pair<double, C_subinterval>(tmp_intersection2, {c_m, c2}));
        break;
      }
    }
    update_epsilon(control_worst_lb_.begin()->first);

    recursive_check_inexact(w_c1, w_cm, grad_w1, grad_w_cm, c1, c_m, epsilon,
                            tmp_intersection1, iter, re_iter);
    recursive_check_inexact(w_cm, w_c2, grad_w_cm, grad_w2, c_m, c2, epsilon,
                            tmp_intersection2, iter, re_iter);
  } else {
    return;
  }
}

void Faop::recursive_check_inexact(
    const double &c1, const std::vector<double> c1_lb_right, const double &c2,
    const std::vector<double> c2_lb_left,
    const std::vector<Eigen::VectorXd> &ws, const double &epsilon,
    const double &the_intersection, int &iter, int &re_iter) {

  if (c2 - c1 <= min_move_c_ || c2 >= c_max_) {
    return;
  }

  double c1_tilde = get_apprx_c_right_inexact(c1, c1_lb_right, epsilon);
  double c2_tilde = get_apprx_c_left_inexact(c2, c2_lb_left, epsilon);
  if (c1_tilde < c2_tilde) {
    std::vector<Eigen::VectorXd> ws_cm;
    std::vector<double> cm_lb_left, cm_lb_right, tmp_c_lb;
    double c_m = 0.5 * (c1_tilde + c2_tilde);
    int num_ub, num_lb, total_num_ub = 0;
    Eigen::VectorXd w_cm;
    for (int i = 0; i < fold_num_; ++i) {
      psol_objs_[i]->set_regularized_parameter(c_m);
      w_cm = psol_objs_[i]->train_warm_start_inexact(ws[i], inexact_level_,
                                                     num_ub, num_lb);
      ws_cm.push_back(w_cm);
      total_num_ub += num_ub;
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_left_subopt(false);
      cm_lb_left.insert(cm_lb_left.end(), tmp_c_lb.begin(), tmp_c_lb.end());
      tmp_c_lb.clear();
      tmp_c_lb = psol_objs_[i]->get_c_set_right_subopt(false);
      cm_lb_right.insert(cm_lb_right.end(), tmp_c_lb.begin(), tmp_c_lb.end());
    }
    update_best(total_num_ub);
    std::sort(cm_lb_left.begin(), cm_lb_left.end(), std::greater<double>());
    std::sort(cm_lb_right.begin(), cm_lb_right.end());
    ++re_iter;
    ++iter;
    // std::cout << iter << " " << c_m << " " << num_ub << " " << c1_tilde << "
    // "
    //           << c2_tilde << " " << 0 << std::endl;
    double tmp_intersection1 = find_worst_lb(c1, c1_lb_right, c_m, cm_lb_left);
    double tmp_intersection2 = find_worst_lb(c_m, cm_lb_right, c2, c2_lb_left);

    auto range = control_worst_lb_.equal_range(the_intersection);
    for (auto it = range.first; it != range.second; ++it) {
      if ((it->second).c_left == c1) {
        control_worst_lb_.erase(it);
        control_worst_lb_.insert(
            std::pair<double, C_subinterval>(tmp_intersection1, {c1, c_m}));
        control_worst_lb_.insert(
            std::pair<double, C_subinterval>(tmp_intersection2, {c_m, c2}));
        break;
      }
    }
    update_epsilon(control_worst_lb_.begin()->first);

    recursive_check_inexact(c1, c1_lb_right, c_m, cm_lb_left, ws_cm, epsilon,
                            tmp_intersection1, iter, re_iter);
    recursive_check_inexact(c_m, cm_lb_right, c2, c2_lb_left, ws_cm, epsilon,
                            tmp_intersection2, iter, re_iter);
  } else {
    return;
  }
}

} // namespace sdm
