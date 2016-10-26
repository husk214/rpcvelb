#include "Validation_error_path.h"

Validation_error_path::Validation_error_path(std::vector<Solver *> train_objs,
                                             const double c_min,
                                             const double c_max,
                                             const double min_move_c)
    : train_obj_(), c_min_(c_min), c_max_(c_max), min_move_c_(min_move_c),
      valid_x_(), valid_y_(), valid_l_(), valid_n_(), valid_x_norm_(),
      best_num_error_(), best_valid_error_(1.0), best_c_(0.0), best_w_(),
      train_objs_(train_objs), fold_num_(), whole_l_(0), num_opti_call_(0),
      total_time_(0.0) {
  // valid_l_ = (train_objs_[0])->get_valid_l();
  // valid_n_ = (train_objs_[0])->get_valid_n();
  fold_num_ = train_objs_.size();
  for (int i = 0; i < fold_num_; ++i)
    whole_l_ += (train_objs_[i])->get_valid_l();
  best_num_error_ = whole_l_;
}

Validation_error_path::Validation_error_path(
    Solver *train_obj, const std::string fileNameLibSVMFormat,
    const double c_min, const double c_max, const double min_move_c)
    : train_obj_(train_obj), c_min_(c_min), c_max_(c_max),
      min_move_c_(min_move_c), valid_x_(), valid_y_(), valid_l_(), valid_n_(),
      valid_x_norm_(), best_num_error_(), best_valid_error_(1.0), best_c_(0.0),
      best_w_(), train_objs_(), fold_num_(), whole_l_(), num_opti_call_(0),
      total_time_(0.0) {
  read_LibSVMdata1((fileNameLibSVMFormat).c_str(), valid_x_, valid_y_);
  valid_l_ = valid_x_.rows();
  valid_n_ = valid_x_.cols();
  valid_x_norm_.resize(valid_l_);
  for (int i = 0; i < valid_l_; ++i)
    valid_x_norm_.coeffRef(i) = (valid_x_.row(i)).norm();
  best_num_error_ = valid_l_;
}

Validation_error_path::Validation_error_path(
    Solver *train_obj, Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x,
    Eigen::ArrayXd valid_y, const double c_min, const double c_max,
    const double min_move_c)
    : train_obj_(train_obj), c_min_(c_min), c_max_(c_max),
      min_move_c_(min_move_c), valid_x_(valid_x), valid_y_(valid_y), valid_l_(),
      valid_n_(), valid_x_norm_(), best_num_error_(), best_valid_error_(1.0),
      best_c_(0.0), best_w_(), train_objs_(), fold_num_(0), whole_l_(0),
      num_opti_call_(0), total_time_(0.0) {
  valid_l_ = valid_x_.rows();
  valid_n_ = valid_x_.cols();
  valid_x_norm_.resize(valid_l_);
  for (int i = 0; i < valid_l_; ++i)
    valid_x_norm_.coeffRef(i) = (valid_x_.row(i)).norm();
  best_num_error_ = valid_l_;
}

Validation_error_path::~Validation_error_path() {}

double Validation_error_path::get_best_c() { return best_c_; }
Eigen::VectorXd Validation_error_path::get_best_w() { return best_w_; }

int Validation_error_path::get_num_opti_call() { return num_opti_call_; }
double Validation_error_path::get_total_time() { return total_time_; }

void Validation_error_path::check_log_scale(const int &num_points,
                                            const double &inexact_level) {

  int n = (train_obj_->get_fun_obj())->get_variable();
  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  Eigen::VectorXd w_star = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd w_tilde = w_star;
  double c_hat, ubve, lbve;
  std::vector<double> star_c_vec, tilde_c_vec;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);
    w_star = train_obj_->train_warm_start(w_star);
    star_c_vec = train_obj_->get_c_set_right_opt(w_star, c_hat);
    std::sort(star_c_vec.begin(), star_c_vec.end());

    w_tilde = train_obj_->train_warm_start_inexact(w_tilde, inexact_level, ubve,
                                                   lbve);
    tilde_c_vec = train_obj_->get_c_set_right_subopt(
        w_tilde, train_obj_->get_grad(), c_hat);
    std::sort(tilde_c_vec.begin(), tilde_c_vec.end());
    int size_diff = star_c_vec.size() - tilde_c_vec.size();

    for (int j = 0; j < static_cast<int>(tilde_c_vec.size()); ++j) {

      if (star_c_vec[j + size_diff] < tilde_c_vec[j])
        std::cout << "check error : " << i << " " << std::setw(7)
                  << std::setprecision(10) << c_hat << " " << std::setw(3) << j
                  << ", " << std::setprecision(10) << star_c_vec[j + size_diff]
                  << " " << std::setprecision(10) << tilde_c_vec[j] << " "
                  << star_c_vec.size() << " " << tilde_c_vec.size() << " "
                  << star_c_vec[j + size_diff] - tilde_c_vec[j] << std::fixed
                  << std::endl;
    }
  }
}

void Validation_error_path::exact_path() {
  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  double valid_err = 0.0;
  int iter_count = 1;
  while (1) {
    w = train_obj_->train_warm_start(w);
    c = get_exact_min_c(w, c, valid_err);
    std::cout << iter_count << " " << c << " " << valid_err << std::endl;
    if (c >= c_max_)
      break;
    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    ++iter_count;
  }
}

void Validation_error_path::exact_best_path() {
  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  double valid_err = 0.0;
  int iter_count = 1;
  while (1) {
    w = train_obj_->train_warm_start(w);
    // std::cout <<predict(w) << std::endl;
    c = get_exact_best_min_c(w, c, valid_err);
    std::cout << iter_count << " " << c << " " << valid_err << std::endl;
    // c += 0.01;
    if (c >= c_max_)
      break;
    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    ++iter_count;
  }
}

void Validation_error_path::exact_path_only_error() {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_, c_old = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  double valid_err = 0.0;
  int iter_count = 1;
  while (1) {
    w = train_obj_->train_warm_start(w);
    // std::cout <<predict(w) << std::endl;
    c = get_exact_c_only_error(w, c, valid_err);
    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_old << " " << valid_err << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;
    if (c >= c_max_)
      break;
    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    c_old = c;
    ++iter_count;
  }
}

void Validation_error_path::approximate_path(const double &epsilon) {

  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_, c_old = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  int iter_count = 0;
  w = train_obj_->train_warm_start(w);
  double valid_err;
  while (1) {
    ++iter_count;
    c = get_approximate_min_c(w, c, epsilon, valid_err);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_old << " " << valid_err << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    w = train_obj_->train_warm_start(w);
    c_old = c;
    if (c >= c_max_)
      break;
  }
}

void Validation_error_path::approximate_best_path(const double &epsilon) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_, c_old = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  int iter_count = 0;
  w = train_obj_->train_warm_start(w);
  double valid_err;
  while (1) {
    ++iter_count;
    c = get_approximate_best_min_c(w, c, epsilon, valid_err);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_old << " " << valid_err << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    w = train_obj_->train_warm_start(w);
    c_old = c;
    if (c >= c_max_)
      break;
  }
}

void Validation_error_path::approximate_path_only_error(const double &epsilon) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_, c_old = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  int iter_count = 0;
  double valid_err;
  w = train_obj_->train_warm_start(w);
  while (1) {
    ++iter_count;
    c = get_approximate_c_only_error(w, c, epsilon, valid_err);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_old << " " << valid_err << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    c_old = c;
    if (c >= c_max_)
      break;
    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    w = train_obj_->train_warm_start(w);
  }
}

void
Validation_error_path::approximate_path_inexact(const double &epsilon,
                                                const double &inexact_level) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_, c_old = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  int iter_count = 0;
  double ub_ve, lb_ve;
  w = train_obj_->train_warm_start_inexact(w, inexact_level, ub_ve, lb_ve);
  while (1) {
    ++iter_count;
    c = get_apprx_path_c_inexact(w, train_obj_->get_grad(), epsilon, c_old);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_old << " " << ub_ve << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    w = train_obj_->train_warm_start_inexact(w, inexact_level, ub_ve, lb_ve);
    c_old = c;
    if (c >= c_max_)
      break;
  }
}

void Validation_error_path::apprx_aggressive(const double &epsilon,
                                             const double &aggressive) {

  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);

  double c = c_min_, c_old = c_min_;
  (train_obj_->get_fun_obj())->set_regularized_parameter(c);
  int iter_count = 0, recu_iter = 0;
  w = train_obj_->train_warm_start(w);
  Eigen::VectorXd w_old = w;
  double valid_err;
  while (1) {
    ++iter_count;
    c = get_approximate_c_only_error(w, c, aggressive, valid_err);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_old << " " << valid_err << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    (train_obj_->get_fun_obj())->set_regularized_parameter(c);
    w = train_obj_->train_warm_start(w);
    recursive_check(w_old, w, c_old, c, epsilon, iter_count, recu_iter);

    c_old = c;
    w_old = w;
    if (c >= c_max_) {
      c = get_approximate_c_only_error(w, c, epsilon, valid_err);
      if (c >= c_max_)
        break;
    }
  }
  std::cout << "recursive iteration " << recu_iter << std::endl;
}

void Validation_error_path::apprx_inexact_train_multi_aggr(
    const int &num_points, const double &epsilon, const double &inexact_level,
    const double &alpha) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  std::vector<Eigen::VectorXd> w_set;
  std::vector<Eigen::VectorXd> grad_set;
  std::vector<double> ub_ve_set;
  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  Eigen::VectorXd w_star = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd grad_w, vw_star, vgrad_w;
  double c_hat, c_hat_old, next_c_hat, ub_c_hat = 1.0, lb_c_hat = 0.0,
                                       tmp_ub_c_hat = 1.0;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);
    w_star = train_obj_->train_warm_start_inexact(w_star, inexact_level,
                                                  tmp_ub_c_hat, lb_c_hat);
    ub_ve_set.push_back(tmp_ub_c_hat);
    w_set.push_back(w_star);
    grad_set.push_back(train_obj_->get_grad());
    int ttt = tmp_ub_c_hat * valid_l_;
    update_best(ttt, tmp_ub_c_hat, c_hat, w_star);
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
    std::cout << iter_count << " " << c_hat << " " << ub_ve_set[i] << " "
              // << train_obj_->get_grad_norm() << " "
              << best_valid_error_ << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << " 0.0" << std::endl;

    while (c_hat < next_c_hat) {
      c_hat = get_apprx_c_right_inexact(w_tilde, grad_w, alpha, c_hat);
      if (c_hat >= next_c_hat) {
        c_hat = get_apprx_c_right_inexact(w_tilde, grad_w, epsilon, c_hat);
        if (c_hat >= next_c_hat)
          break;
      }
      (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);

      w_tilde = train_obj_->train_warm_start_inexact(w_tilde, inexact_level,
                                                     ub_c_hat, lb_c_hat);
      grad_w = train_obj_->get_grad();
      int ttt = ub_c_hat * valid_l_;
      update_best(ttt, ub_c_hat, c_hat, w_tilde);
      grad_w = train_obj_->get_grad();
      if (alpha > epsilon)
        recursive_check_inexact(w_tilde_old, w_tilde, grad_w_old, grad_w,
                                c_hat_old, c_hat, epsilon, inexact_level,
                                iter_count, recu_iter);
      ++iter_count;
      end = std::chrono::system_clock::now();
      diff = end - start;
      std::cout << iter_count << " " << c_hat << " " << ub_c_hat << " "
                // << train_obj_->get_grad_norm() << " "
                << best_valid_error_ << " "
                << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                        .count() /
                    1000.0) << " " << ub_c_hat - lb_c_hat << std::endl;
      c_hat_old = c_hat;
      w_tilde_old.noalias() = w_tilde;
      grad_w_old.noalias() = grad_w;
    }
    num_opti_call_ = iter_count;
    total_time_ =
        (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() /
         1000.0);
  }
  // std::cout << '\n' << "total iter: " << iter_count
  //           << ", recursive_iter: " << recu_iter
  //           << ", best ve: " << best_valid_error_ << ", c:" << best_c_
  //           << ", time: "
  //           << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
  //                   .count() /
  //               1000.0) << std::endl;
}

void
Validation_error_path::approximate_inexact_train(const double &epsilon,
                                                 const double &inexact_level) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  double now_c = c_min_;
  double ub_c_now, lb_c_now;
  int n = (train_obj_->get_fun_obj())->get_variable();
  Eigen::VectorXd w_now_c = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd vw_now_c, grad_w_now_c;
  int iter_count = 1;
  int th = 0;
  while (1) {
    (train_obj_->get_fun_obj())->set_regularized_parameter(now_c);
    w_now_c = train_obj_->train_warm_start_inexact(w_now_c, inexact_level,
                                                   ub_c_now, lb_c_now);
    int ttt = ub_c_now * valid_l_;
    update_best(ttt, ub_c_now, now_c);

    vw_now_c = w_now_c;
    grad_w_now_c = train_obj_->get_grad();
    if (vw_now_c.size() != valid_n_) {
      vw_now_c.conservativeResize(valid_n_);
      grad_w_now_c.conservativeResize(valid_n_);
    }

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << now_c << " " << ub_c_now << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << " " << th << std::endl;

    now_c =
        get_apprx_c_right_inexact(vw_now_c, grad_w_now_c, epsilon, now_c, th);
    if (now_c >= c_max_)
      break;

    ++iter_count;
  }
}

void Validation_error_path::multi_exact(const int &num_points) {
  multi_apprx(num_points, 0.0);
}

void Validation_error_path::multi_apprx(const int &num_points,
                                        const double &epsilon) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  std::vector<Eigen::VectorXd> w_set;
  std::vector<Eigen::VectorXd> grad_set;
  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  Eigen::VectorXd w_star = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd grad_w, tmp_w;
  double c_hat, ve_c_hat;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);
    w_star = train_obj_->train_warm_start(w_star);
    grad_w = train_obj_->get_grad();

    w_set.push_back(w_star);

    grad_set.push_back(grad_w);

    ve_c_hat = train_obj_->get_valid_error(w_star);
    int tmp_num_err = ve_c_hat * valid_l_;
    update_best(tmp_num_err, ve_c_hat, c_hat);
  }

  int iter_count = 0;
  double next_c_hat;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    next_c_hat = pow(10.0, (log_cmin + (i + 1.0) * log_interval));
    w_star = w_set[i];

    ++iter_count;

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_hat << " " << ve_c_hat << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;
    c_hat = get_approximate_c_only_error(w_star, c_hat, epsilon, ve_c_hat);
    while (c_hat < next_c_hat) {
      (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);
      w_star = train_obj_->train_warm_start(w_star);

      end = std::chrono::system_clock::now();
      diff = end - start;
      ++iter_count;
      std::cout << iter_count << " " << c_hat << " " << ve_c_hat << " "
                << train_obj_->get_grad_norm() << " "
                << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                        .count() /
                    1000.0) << std::endl;
      c_hat = get_approximate_c_only_error(w_star, c_hat, epsilon, ve_c_hat);
    }
  }
}

void Validation_error_path::multi_exact_inexact_train(const int &num_points) {
  double ie_lv = 0.5 / valid_l_;
  multi_apprx_inexact_train(num_points, 1.0 / valid_l_, ie_lv, min_move_c_);
}
void Validation_error_path::multi_apprx_inexact_train(
    const int &num_points, const double &epsilon, const double &inexact_level,
    const double &accuracy_binary_search) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;

  int n = (train_obj_->get_fun_obj())->get_variable();
  std::vector<Eigen::VectorXd> w_set;
  std::vector<Eigen::VectorXd> grad_set;
  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  Eigen::VectorXd w_star = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd grad_w, tmp_w;
  double c_hat, ve_c_hat;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);
    w_star = train_obj_->train_warm_start(w_star);
    grad_w = train_obj_->get_grad();
    w_set.push_back(w_star);
    grad_set.push_back(grad_w);

    ve_c_hat = get_valid_error(w_star);
    update_best(ve_c_hat, c_hat);
  }

  int iter_count = 0;
  double next_c_hat, ub_c_hat = 1.0, lb_c_hat = 0.0;
  Eigen::VectorXd w_tilde;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    next_c_hat = pow(10.0, (log_cmin + (i + 1.0) * log_interval));
    w_tilde = w_set[i];
    ++iter_count;

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_hat << " " << ub_c_hat << " "
              << train_obj_->get_grad_norm() << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << " 0.0" << std::endl;
    c_hat = get_approximate_c_only_error(w_tilde, c_hat, epsilon, ve_c_hat);

    while (c_hat < next_c_hat) {
      (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);
      w_tilde = train_obj_->train_warm_start_inexact(w_tilde, inexact_level,
                                                     ub_c_hat, lb_c_hat);
      update_best(ub_c_hat, c_hat);
      grad_w = train_obj_->get_grad();
      if (w_tilde.size() != valid_n_) {
        tmp_w = w_tilde;
        tmp_w.conservativeResize(valid_n_);
        grad_w.conservativeResize(valid_n_);
        end = std::chrono::system_clock::now();
        diff = end - start;
        ++iter_count;
        std::cout << iter_count << " " << c_hat << " " << ub_c_hat << " "
                  << train_obj_->get_grad_norm() << " "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(
                          diff).count() /
                      1000.0) << " " << ub_c_hat - lb_c_hat << std::endl;

        c_hat = get_apprx_c_bisec_right(tmp_w, grad_w, epsilon, c_hat);
      } else {
        // ub_c_hat = get_upper_bound_valid_error(w_tilde, grad_w);
        end = std::chrono::system_clock::now();
        diff = end - start;
        ++iter_count;
        std::cout << iter_count << " " << c_hat << " " << ub_c_hat << " "
                  << train_obj_->get_grad_norm() << " "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(
                          diff).count() /
                      1000.0) << " " << ub_c_hat - lb_c_hat << std::endl;

        c_hat = get_apprx_c_bisec_right(w_tilde, grad_w, epsilon, c_hat);
      }
    }
  }
}

void
Validation_error_path::cross_validation_apprx_exact(const double &epsilon) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  double now_c = c_min_;
  double valid_error;
  int n = ((train_objs_[0])->get_fun_obj())->get_variable();
  Eigen::VectorXd w_now_c = Eigen::VectorXd::Zero(n);
  int iter_count = 1;
  std::vector<Eigen::VectorXd> w_set;
  while (1) {
    int ttt = 0, tt;
    w_set.clear();
    for (auto sol : train_objs_) {
      (sol->get_fun_obj())->set_regularized_parameter(now_c);
      sol->train_warm_start(sol->get_w());
      w_set.push_back(sol->get_w());
      tt = sol->get_valid_error(sol->get_w()) * (sol->get_valid_l());
      ttt += tt;
    }
    valid_error = (double)ttt / whole_l_;

    update_best(ttt, valid_error, now_c);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << now_c << " " << valid_error << " "
              << best_valid_error_ << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    now_c = get_apprx_c_right_exact_cv(w_set, epsilon, now_c);
    if (now_c >= c_max_)
      break;

    ++iter_count;
  }
}

void Validation_error_path::cross_validation_apprx_exact_multi(
    const int &num_points, const double &epsilon) {

  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  std::vector<Eigen::VectorXd> w_set, pre_w_set, grad_w_set, pre_grad_w_set;

  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  std::vector<std::vector<Eigen::VectorXd>> multi_w_sets, multi_graw_w_sets;
  double c_hat, next_c_hat, valid_error;
  std::vector<double> valid_error_set;
  int ttt, tt;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    w_set.clear();
    ttt = 0;
    for (int i = 0; i < fold_num_; ++i) {
      ((train_objs_[i])->get_fun_obj())->set_regularized_parameter(c_hat);
      (train_objs_[i])->train_warm_start((train_objs_[i])->get_w());
      tt = (train_objs_[i])->get_valid_error((train_objs_[i])->get_w()) *
           (train_objs_[i])->get_valid_l();
      ttt += tt;
      w_set.push_back(train_objs_[i]->get_w());
    }
    multi_w_sets.push_back(w_set);
    multi_graw_w_sets.push_back(grad_w_set);
    valid_error = (double)ttt / whole_l_;
    valid_error_set.push_back(valid_error);
    update_best(ttt, valid_error, c_hat);
  }

  int iter_count = 0;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    next_c_hat = pow(10.0, (log_cmin + (i + 1.0) * log_interval));
    w_set.clear();
    std::copy((multi_w_sets[i]).begin(), (multi_w_sets[i]).end(),
              std::back_inserter(w_set));

    ++iter_count;
    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_hat << " " << valid_error_set[i] << " "
              << best_valid_error_ << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << " 0.0" << std::endl;
    while (c_hat < next_c_hat) {
      c_hat = get_apprx_c_right_exact_cv(w_set, epsilon, c_hat);
      if (c_hat >= next_c_hat)
        break;

      ttt = 0;
      w_set.clear();
      for (int j = 0; j < fold_num_; ++j) {
        ((train_objs_[j])->get_fun_obj())->set_regularized_parameter(c_hat);
        (train_objs_[j])->train_warm_start((train_objs_[j])->get_w());

        w_set.push_back((train_objs_[j])->get_w());
        tt = (train_objs_[j])->get_valid_error(train_objs_[j]->get_w()) *
             (train_objs_[j])->get_valid_l();
        ttt += tt;
      }
      valid_error = (double)ttt / whole_l_;
      update_best(ttt, valid_error, c_hat);
      ++iter_count;
      end = std::chrono::system_clock::now();
      diff = end - start;
      std::cout << iter_count << " " << c_hat << " " << valid_error << " "
                << best_valid_error_ << " "
                << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                        .count() /
                    1000.0) << " "
                << "0.0" << std::endl;
    }
  }
}

void Validation_error_path::cross_validation_apprx_inexact(
    const double &epsilon, const double &inexact_level) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  double now_c = c_min_;
  double mean_ub_c_now, ub_c_now, lb_c_now;
  int n = ((train_objs_[0])->get_fun_obj())->get_variable();
  Eigen::VectorXd w_now_c = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd vw_now_c, grad_w_now_c;
  int iter_count = 1;
  std::vector<Eigen::VectorXd> w_set, grad_w_set;
  while (1) {
    int ttt = 0, tt;
    w_set.clear();
    grad_w_set.clear();
    for (auto sol : train_objs_) {
      (sol->get_fun_obj())->set_regularized_parameter(now_c);
      sol->train_warm_start_inexact(sol->get_w(), inexact_level, ub_c_now,
                                    lb_c_now);
      w_set.push_back(sol->get_w());
      grad_w_set.push_back(sol->get_grad());
      tt = ub_c_now * (sol->get_valid_l());
      ttt += tt;
    }
    mean_ub_c_now = (double)ttt / whole_l_;

    update_best(ttt, mean_ub_c_now, now_c);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << now_c << " " << mean_ub_c_now << " "
              << best_valid_error_ << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    now_c = get_apprx_c_right_inexact_cv(w_set, grad_w_set, epsilon, now_c);
    if (now_c >= c_max_)
      break;

    ++iter_count;
  }
}

void Validation_error_path::cross_validation_apprx_inexact_multi_aggr(
    const int &num_points, const double &epsilon, const double &inexact_level,
    const double &aggressive) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  std::vector<Eigen::VectorXd> w_set, pre_w_set, grad_w_set, pre_grad_w_set;

  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / num_points;

  std::vector<std::vector<Eigen::VectorXd>> multi_w_sets, multi_graw_w_sets;
  double c_hat, c_hat_old, next_c_hat, c_hat_mem, ub_c_hat = 1.0,
                                                  lb_c_hat = 0.0, mean_ub_c_hat;
  std::vector<double> mean_ub_set;
  int ttt, tt;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    w_set.clear();
    grad_w_set.clear();
    ttt = 0;
    for (int i = 0; i < fold_num_; ++i) {
      ((train_objs_[i])->get_fun_obj())->set_regularized_parameter(c_hat);
      (train_objs_[i])->train_warm_start_inexact(
          (train_objs_[i])->get_w(), inexact_level, ub_c_hat, lb_c_hat);
      tt = ub_c_hat * (train_objs_[i])->get_valid_l();
      ttt += tt;
      w_set.push_back(train_objs_[i]->get_w());
      grad_w_set.push_back(train_objs_[i]->get_grad());
    }
    multi_w_sets.push_back(w_set);
    multi_graw_w_sets.push_back(grad_w_set);
    mean_ub_c_hat = (double)ttt / whole_l_;
    mean_ub_set.push_back(mean_ub_c_hat);
    update_best(ttt, mean_ub_c_hat, c_hat);
  }
  // std::cout <<"pre recursive_check  " << w_set.size() <<std::endl;

  int iter_count = 0, recu_iter = 0;
  for (int i = 0; i < num_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    c_hat_old = c_hat;
    next_c_hat = pow(10.0, (log_cmin + (i + 1.0) * log_interval));
    w_set.clear();
    grad_w_set.clear();
    pre_w_set.clear();
    pre_grad_w_set.clear();
    std::copy((multi_w_sets[i]).begin(), (multi_w_sets[i]).end(),
              std::back_inserter(w_set));
    std::copy((multi_w_sets[i]).begin(), (multi_w_sets[i]).end(),
              std::back_inserter(pre_w_set));
    std::copy((multi_graw_w_sets[i]).begin(), (multi_graw_w_sets[i]).end(),
              std::back_inserter(grad_w_set));
    std::copy((multi_graw_w_sets[i]).begin(), (multi_graw_w_sets[i]).end(),
              std::back_inserter(pre_grad_w_set));

    ++iter_count;
    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << c_hat << " " << mean_ub_set[i] << " "
              << best_valid_error_ << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << " 0.0" << std::endl;
    while (c_hat < next_c_hat) {
      c_hat_mem = c_hat;
      c_hat = get_apprx_c_right_inexact_cv(pre_w_set, pre_grad_w_set,
                                           aggressive, c_hat_mem);
      if (c_hat >= next_c_hat) {
        c_hat = get_apprx_c_right_inexact_cv(pre_w_set, pre_grad_w_set, epsilon,
                                             c_hat_mem);
        if (c_hat >= next_c_hat) {
          break;
        }
      }

      ttt = 0;
      w_set.clear();
      grad_w_set.clear();
      for (int j = 0; j < fold_num_; ++j) {
        ((train_objs_[j])->get_fun_obj())->set_regularized_parameter(c_hat);
        (train_objs_[j])->train_warm_start_inexact(
            (train_objs_[j])->get_w(), inexact_level, ub_c_hat, lb_c_hat);

        w_set.push_back((train_objs_[j])->get_w());
        grad_w_set.push_back((train_objs_[j])->get_grad());
        tt = ub_c_hat * (train_objs_[j])->get_valid_l();
        ttt += tt;
      }
      mean_ub_c_hat = (double)ttt / whole_l_;
      update_best(ttt, mean_ub_c_hat, c_hat);
      recursive_check_inexact_cv(pre_w_set, w_set, pre_grad_w_set, grad_w_set,
                                 c_hat_old, c_hat, epsilon, inexact_level,
                                 iter_count, recu_iter);
      ++iter_count;
      end = std::chrono::system_clock::now();
      diff = end - start;
      std::cout << iter_count << " " << c_hat << " " << mean_ub_c_hat << " "
                << best_valid_error_ << " "
                << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                        .count() /
                    1000.0) << " " << ub_c_hat - lb_c_hat << std::endl;
      c_hat_old = c_hat;
      pre_w_set.clear();
      pre_grad_w_set.clear();
      std::copy(w_set.begin(), w_set.end(), std::back_inserter(pre_w_set));
      std::copy(grad_w_set.begin(), grad_w_set.end(),
                std::back_inserter(pre_grad_w_set));
    }
  }
  // std::cout << '\n' << "total iter: " << iter_count
  //           << ", recursive iter: " << recu_iter
  //           << ", best ve: " << best_valid_error_ << ", c: " << best_c_
  //           << ", time: "
  //           << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
  //                   .count() /
  //               1000.0) << std::endl;
}

void Validation_error_path::cross_validation_apprx_exact_path(
    const double &epsilon) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  double now_c = c_min_;
  double valid_error;
  int n = ((train_objs_[0])->get_fun_obj())->get_variable();
  Eigen::VectorXd w_now_c = Eigen::VectorXd::Zero(n);
  int iter_count = 1;
  std::vector<Eigen::VectorXd> w_set;
  while (1) {
    int ttt = 0, tt;
    w_set.clear();
    for (auto sol : train_objs_) {
      (sol->get_fun_obj())->set_regularized_parameter(now_c);
      sol->train_warm_start(sol->get_w());
      w_set.push_back(sol->get_w());
      tt = sol->get_valid_error(sol->get_w()) * (sol->get_valid_l());
      ttt += tt;
    }
    valid_error = (double)ttt / whole_l_;

    update_best(ttt, valid_error, now_c);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << now_c << " " << valid_error << " "
              << best_valid_error_ << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    now_c = get_apprx_c_right_exact_cv_for_path(w_set, epsilon, now_c);
    if (now_c >= c_max_)
      break;

    ++iter_count;
  }
}

void Validation_error_path::cross_validation_apprx_inexact_path(
    const double &epsilon, const double &inexact_level) {
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  double now_c = c_min_;
  double mean_ub_c_now, ub_c_now, lb_c_now;
  int n = ((train_objs_[0])->get_fun_obj())->get_variable();
  Eigen::VectorXd w_now_c = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd vw_now_c, grad_w_now_c;
  int iter_count = 1;
  std::vector<Eigen::VectorXd> w_set, grad_w_set;
  while (1) {
    int ttt = 0, tt;
    double dd =0;
    w_set.clear();
    grad_w_set.clear();
    for (auto sol : train_objs_) {
      (sol->get_fun_obj())->set_regularized_parameter(now_c);
      sol->train_warm_start_inexact(sol->get_w(), inexact_level, ub_c_now,
                                    lb_c_now);
      // std::cout <<"vep ubve " << ub_c_now <<" lbve " <<lb_c_now <<" d " <<ub_c_now -lb_c_now <<std::endl;
      dd += ub_c_now - lb_c_now;
      w_set.push_back(sol->get_w());
      grad_w_set.push_back(sol->get_grad());
      tt = ub_c_now * (sol->get_valid_l());
      ttt += tt;
    }
    // std::cout <<" d t " <<(double)dd/fold_num_ <<std::endl;
    mean_ub_c_now = (double)ttt / whole_l_;

    update_best(ttt, mean_ub_c_now, now_c);

    end = std::chrono::system_clock::now();
    diff = end - start;
    std::cout << iter_count << " " << now_c << " " << mean_ub_c_now << " "
              << best_valid_error_ << " "
              << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
                      .count() /
                  1000.0) << std::endl;

    now_c = get_apprx_c_right_inexact_cv_for_path(w_set, grad_w_set, epsilon,
                                                  now_c);
    if (now_c >= c_max_)
      break;

    ++iter_count;
  }
}

void Validation_error_path::grid_search_log_scale(const int &num_grid_points) {
  clear_best();
  double log_cmin = log10(c_min_);
  double log_interval = (log10(c_max_) - log_cmin) / (num_grid_points - 1);
  double c_hat, ve_c_hat;
  Eigen::VectorXd w_star = Eigen::VectorXd::Zero(valid_n_);
  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  int ttt = 0;
  for (int i = 0; i < num_grid_points; ++i) {
    c_hat = pow(10.0, (log_cmin + i * log_interval));
    (train_obj_->get_fun_obj())->set_regularized_parameter(c_hat);
    w_star = train_obj_->train_warm_start(w_star);
    ve_c_hat = get_valid_error(w_star);
    ttt = ve_c_hat * valid_l_;
    update_best(ttt, ve_c_hat, c_hat, w_star);
    end = std::chrono::system_clock::now();
    diff = end - start;
    // std::cout << i + 1 << " " << c_hat << " " << ve_c_hat << " " <<ttt <<" "
    //           << (std::chrono::duration_cast<std::chrono::milliseconds>(diff)
    //                   .count() /
    //               1000.0) << std::endl;
  }
  num_opti_call_ = num_grid_points;
  total_time_ =
      (std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() /
       1000.0);
}

void Validation_error_path::clear_best() {
  best_num_error_ = valid_l_;
  best_valid_error_ = 1.0;
  best_c_ = 0.0;
  best_w_ = Eigen::VectorXd::Zero(valid_n_);
}

void Validation_error_path::update_best(const double tmp_valid_err,
                                        const double tmp_c) {
  if (best_valid_error_ > tmp_valid_err) {
    best_valid_error_ = tmp_valid_err;
    best_c_ = tmp_c;
  }
}

void Validation_error_path::update_best(const int tmp_num_err,
                                        const double tmp_valid_err,
                                        const double tmp_c) {
  if (best_num_error_ > tmp_num_err) {
    best_num_error_ = tmp_num_err;
    best_valid_error_ = tmp_valid_err;
    best_c_ = tmp_c;
  }
}

void Validation_error_path::update_best(const int tmp_num_err,
                                        const double tmp_valid_err,
                                        const double tmp_c,
                                        const Eigen::VectorXd &tmp_w) {
  if (best_num_error_ > tmp_num_err) {
    best_num_error_ = tmp_num_err;
    best_valid_error_ = tmp_valid_err;
    best_c_ = tmp_c;
    best_w_.noalias() = tmp_w;
  }
}

double Validation_error_path::get_valid_error(const Eigen::VectorXd &w) {
  int miss = 0;
  Eigen::VectorXd vw = w;
  if (valid_n_ != w.size()) {
    vw.conservativeResize(valid_n_);
  }
  Eigen::ArrayXd prob = valid_y_ * (valid_x_ * vw).array();

  for (int i = 0; i < valid_l_; ++i) {
    if (prob.coeffRef(i) < 0.0)
      ++miss;
  }

  return (double)miss / valid_l_;
}

double Validation_error_path::get_exact_min_c(const Eigen::VectorXd &w,
                                              const double &now_c,
                                              double &valid_error) {
  valid_error = 0.0;
  Matrix<
  Eigen::VectorXd vw = w;
  if (w.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();
  double vw_norm = vw.norm();
  double xi_norm;
  double zi;
  double min_c = 1e+32, tmp_c;
  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0)
      ++valid_error;

    if (zi < 0.0) {
      tmp_c = (vw_norm * xi_norm - zi) * now_c / (vw_norm * xi_norm + zi);
    } else if (zi > 0.0) {
      tmp_c = (vw_norm * xi_norm + zi) * now_c / (vw_norm * xi_norm - zi);
    } else
      tmp_c = min_move_c_;

    if (min_c > tmp_c) {
      min_c = tmp_c;
    }
  }
  if (min_c - now_c < min_move_c_)
    min_c = now_c + min_move_c_;
  valid_error /= valid_l_;
  update_best(valid_l_, valid_error, min_c);
  return min_c;
}

double Validation_error_path::get_exact_best_min_c(const Eigen::VectorXd &w,
                                                   const double &now_c,
                                                   double &valid_error) {
  valid_error = 0.0;
  Eigen::VectorXd vw = w;
  if (w.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();
  double vw_norm = vw.norm();
  double xi_norm;
  double zi;
  double min_c = c_max_, tmp_c;
  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0) {
      ++valid_error;

      if (zi < 0.0) {
        tmp_c = (vw_norm * xi_norm - zi) * now_c / (vw_norm * xi_norm + zi);
      } else {
        tmp_c = (vw_norm * xi_norm + zi) * now_c / (vw_norm * xi_norm - zi);
      }
      if (min_c > tmp_c) {
        min_c = tmp_c;
      }
    }
  }
  if (min_c - now_c < min_move_c_)
    min_c = now_c + min_move_c_;

  double ve_tmp = (double)valid_error / valid_l_;
  update_best(valid_error, ve_tmp, now_c, w);
  return min_c;
}

double Validation_error_path::get_exact_c_only_error(const Eigen::VectorXd &w,
                                                     const double &now_c,
                                                     double &valid_error) {
  Eigen::VectorXd vw = w;
  if (w.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();
  double vw_norm = vw.norm();
  double xi_norm;
  double zi;
  double tmp_c;
  std::vector<double> c_vec;
  int tmp_num_err = 0;
  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0) {
      ++tmp_num_err;
      if (zi < 0.0) {
        tmp_c = (vw_norm * xi_norm - zi) * now_c / (vw_norm * xi_norm + zi);
      } else if (zi > 0.0) {
        tmp_c = (vw_norm * xi_norm + zi) * now_c / (vw_norm * xi_norm - zi);
      }
      c_vec.push_back(tmp_c);
    }
  }
  valid_error = (double)tmp_num_err / valid_l_;
  update_best(tmp_num_err, valid_error, now_c);

  std::sort(c_vec.begin(), c_vec.end());
  double min_c = now_c + min_move_c_;
  int th = tmp_num_err - best_num_error_;
  if (c_vec.empty())
    return c_max_;
  if (th < 1) {
    min_c = c_vec.at(0);
  } else if (th <= static_cast<int>(c_vec.size()) - 1) {
    min_c = c_vec.at(th);
  } else {
    min_c = c_vec.at(c_vec.size() - 1);
  }
  if (min_c - now_c < min_move_c_)
    min_c = now_c + min_move_c_;

  return min_c;
}

double Validation_error_path::get_approximate_min_c(const Eigen::VectorXd &w,
                                                    const double &now_c,
                                                    const double &eps,
                                                    double &valid_error) {
  Eigen::VectorXd vw = w;
  if (w.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();

  int eps_instance = valid_l_ * eps;
  // if (eps_instance <= 1) {
  //   return get_exact_min_c(w, now_c, valid_error);
  // }
  std::vector<double> c_vec;
  double vw_norm = vw.norm();
  double xi_norm;
  double zi;
  double tmp_c;
  valid_error = 0.0;

  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0) {
      ++valid_error;
    }
    if (zi < 0.0) {
      tmp_c = (vw_norm * xi_norm - zi) * now_c / (vw_norm * xi_norm + zi);
    } else {
      tmp_c = (vw_norm * xi_norm + zi) * now_c / (vw_norm * xi_norm - zi);
    }
    c_vec.push_back(tmp_c);
  }

  std::sort(c_vec.begin(), c_vec.end());
  double min_c = c_vec.at(eps_instance);
  if (min_c - now_c < min_move_c_)
    min_c = now_c + min_move_c_;
  valid_error /= valid_l_;
  return min_c;
}

double Validation_error_path::get_approximate_best_min_c(
    const Eigen::VectorXd &w, const double &now_c, const double &eps,
    double &valid_error) {
  Eigen::VectorXd vw = w;
  if (w.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();

  int eps_instance = valid_l_ * eps;
  if (eps_instance <= 1) {
    return get_exact_best_min_c(w, now_c, valid_error);
  }
  std::vector<double> c_vec;
  double vw_norm = vw.norm();
  double xi_norm;
  double zi;
  double tmp_c;
  valid_error = 0.0;

  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0) {
      ++valid_error;
      if (zi < 0.0) {
        tmp_c = (vw_norm * xi_norm - zi) * now_c / (vw_norm * xi_norm + zi);
      } else if (zi > 0.0) {
        tmp_c = (vw_norm * xi_norm + zi) * now_c / (vw_norm * xi_norm - zi);
      } else {
        tmp_c = min_move_c_;
      }
      if (static_cast<int>(c_vec.size()) >= eps_instance) {
        std::sort(c_vec.begin(), c_vec.end(), std::greater<double>());
        if (c_vec[0] > tmp_c) {
          c_vec.erase(c_vec.begin());
          c_vec.push_back(tmp_c);
        }
      } else {
        c_vec.push_back(tmp_c);
      }
    }
  }

  std::sort(c_vec.begin(), c_vec.end(), std::greater<double>());
  double min_c = c_vec[0];
  if (min_c - now_c < min_move_c_)
    min_c = now_c + min_move_c_;
  valid_error /= valid_l_;
  return min_c;
}

double Validation_error_path::get_approximate_c_only_error(
    const Eigen::VectorXd &w, const double &now_c, const double &tolerance,
    double &valid_error) {
  Eigen::VectorXd vw = w;
  if (w.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();

  int tolerance_instance = valid_l_ * tolerance;
  // if (tolerance_instance <= 1) {
  //   return get_exact_c_only_error(w, now_c, valid_error);
  // }
  std::vector<double> c_vec;
  double vw_norm = vw.norm();
  double xi_norm;
  double zi;
  int tmp_num_err = 0;

  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0) {
      ++tmp_num_err;
      if (zi < 0.0) {
        c_vec.push_back((vw_norm * xi_norm - zi) * now_c /
                        (vw_norm * xi_norm + zi));
      } else if (zi > 0.0) {
        c_vec.push_back((vw_norm * xi_norm + zi) * now_c /
                        (vw_norm * xi_norm - zi));
      }
    }
  }
  valid_error = (double)tmp_num_err / valid_l_;
  update_best(tmp_num_err, valid_error, now_c);

  std::sort(c_vec.begin(), c_vec.end());
  double min_c;
  int th = tmp_num_err - best_num_error_ + tolerance_instance;
  if (c_vec.empty())
    return c_max_;
  if (th < 1) {
    min_c = c_vec.at(0);
  } else if (th <= static_cast<int>(c_vec.size()) - 1) {
    min_c = c_vec.at(th);
  } else {
    min_c = c_vec.at(c_vec.size() - 1);
  }
  if (min_c - now_c < min_move_c_)
    min_c = now_c + min_move_c_;

  return min_c;
}

double Validation_error_path::get_apprx_c_prev(const Eigen::VectorXd &w_star,
                                               const double &now_c,
                                               const double &eps,
                                               double &valid_error) {
  Eigen::VectorXd vw = w_star;
  if (w_star.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();

  int eps_instance = valid_l_ * eps;
  if (eps_instance < 1)
    eps_instance = 1;

  std::vector<double> c_vec;
  double vw_norm = vw.norm();
  double xi_norm;
  double zi;
  double tmp_c = 0.0;
  int tmp_num_err = 0;

  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0) {
      ++tmp_num_err;
      if (zi < 0.0) {
        tmp_c = (vw_norm * xi_norm + zi) * now_c / (vw_norm * xi_norm - zi);
      } else if (zi > 0.0) {
        tmp_c = (vw_norm * xi_norm - zi) * now_c / (vw_norm * xi_norm + zi);
      } else {
        tmp_c = min_move_c_;
      }
      c_vec.push_back(tmp_c);
    }
  }

  std::sort(c_vec.begin(), c_vec.end(), std::greater<double>());
  double max_c;
  int th = tmp_num_err - best_num_error_ + eps_instance;
  if (c_vec.empty())
    return c_max_;
  if (th < 1) {
    max_c = c_vec.at(0);
  } else if (th <= static_cast<int>(c_vec.size()) - 1) {
    max_c = c_vec.at(th);
  } else {
    max_c = c_vec.at(c_vec.size() - 1);
  }

  if (now_c - max_c < min_move_c_)
    max_c = now_c - min_move_c_;
  valid_error = (double)tmp_num_err / valid_l_;

  update_best(tmp_num_err, valid_error, max_c);
  return max_c;
}

double Validation_error_path::get_apprx_c_bisec_right(
    const Eigen::VectorXd &w, const Eigen::VectorXd &grad_w,
    const double &tolerance, const double &now_c, const double &accuracy) {
  Eigen::VectorXd loss_grad = (grad_w - w) / now_c;

  double c1 = now_c;
  double c2 = c_max_;
  double c_m = 0.0;
  double lb_ve_cm;
  double limit_lb = best_valid_error_ - tolerance;
  if (limit_lb <= 0.0)
    limit_lb = DBL_MIN;
  lb_ve_cm = get_lower_bound_valid_error(w, c2 * loss_grad);
  if (lb_ve_cm > limit_lb)
    return c2;

  double c_max_check = 2.0 * c1;
  while (c_max_check < c_max_) {
    lb_ve_cm = get_lower_bound_valid_error(w, c_max_check * loss_grad);
    if ((lb_ve_cm > limit_lb) || (2.0 * c_max_check > c_max_)) {
      c_max_check = c_max_;
      break;
    } else {
      c_max_check *= 2.0;
    }
  }
  c2 = c_max_check;
  while (1) {
    c_m = 0.5 * (c1 + c2);
    lb_ve_cm = get_lower_bound_valid_error(w, c_m * loss_grad);
    if (lb_ve_cm > limit_lb) {
      c1 = c_m;
      if (c2 - c1 <= accuracy) {
        c_m = c1;
        break;
      }
    } else if (lb_ve_cm < limit_lb) {
      c2 = c_m;
      if (c2 - c1 <= accuracy) {
        c_m = c2;
        break;
      }
    } else {
      break;
    }
  }
  if (c_m - now_c < min_move_c_)
    c_m = now_c + min_move_c_;

  return c_m;
}

double Validation_error_path::get_apprx_c_bisec_right_cv(
    const std::vector<Eigen::VectorXd> &w_set,
    const std::vector<Eigen::VectorXd> &grad_w_set, const double &tolerance,
    const double &now_c, const double &accuracy) {
  double c1 = now_c;
  double c2 = c_max_;
  double c_m = 0.0;
  double mean_lb_ve_cm = 0.0;
  double limit_lb = best_valid_error_ - tolerance;
  if (limit_lb <= 0.0)
    limit_lb = DBL_MIN;
  Eigen::VectorXd tmp_c_grad;
  for (int i = 0; i < fold_num_; ++i)
    mean_lb_ve_cm += (train_objs_[i])->get_lower_bound_valid_error(
        w_set[i], grad_w_set[i], now_c, c2);

  mean_lb_ve_cm /= fold_num_;

  if (mean_lb_ve_cm > limit_lb)
    return c2;

  double c_max_check = 2.0 * c1;
  while (c_max_check < c_max_) {
    mean_lb_ve_cm = 0.0;
    for (int i = 0; i < fold_num_; ++i)
      mean_lb_ve_cm += (train_objs_[i])->get_lower_bound_valid_error(
          w_set[i], grad_w_set[i], now_c, c_max_check);
    mean_lb_ve_cm /= fold_num_;
    if ((mean_lb_ve_cm > limit_lb) || (2.0 * c_max_check > c_max_)) {
      c_max_check = c_max_;
      break;
    } else {
      c_max_check *= 2.0;
    }
  }

  c2 = c_max_check;
  while (1) {
    c_m = 0.5 * (c1 + c2);
    mean_lb_ve_cm = 0.0;
    for (int i = 0; i < fold_num_; ++i)
      mean_lb_ve_cm += (train_objs_[i])->get_lower_bound_valid_error(
          w_set[i], grad_w_set[i], now_c, c2);
    mean_lb_ve_cm /= fold_num_;

    if (mean_lb_ve_cm > limit_lb) {
      c1 = c_m;
      if (c2 - c1 <= accuracy) {
        c_m = c1;
        break;
      }
    } else if (mean_lb_ve_cm < limit_lb) {
      c2 = c_m;
      if (c2 - c1 <= accuracy) {
        c_m = c2;
        break;
      }
    } else {
      break;
    }
  }
  if (c_m - now_c < min_move_c_)
    c_m = now_c + min_move_c_;
  return c_m;
}

double Validation_error_path::get_apprx_c_bisec_left(
    const Eigen::VectorXd &w, const Eigen::VectorXd &grad_w,
    const double &tolerance, const double &now_c, const double &pre_c,
    const double &accuracy) {
  Eigen::VectorXd loss_grad = (grad_w - w) / now_c;

  double c1 = now_c;
  double c2 = pre_c;
  double c_m, lb_ve_cm;
  double limit_lb = best_valid_error_ - tolerance;
  if (limit_lb <= 0.0)
    limit_lb = DBL_MIN;

  while (1) {
    c_m = 0.5 * (c1 + c2);
    lb_ve_cm = get_lower_bound_valid_error(w, c_m * loss_grad);

    if (lb_ve_cm > limit_lb) {
      c1 = c_m;
      if (c1 - c2 <= accuracy) {
        c_m = c1;
        break;
      }
    } else if (lb_ve_cm < limit_lb) {
      c2 = c_m;
      if (c1 - c2 <= accuracy) {
        c_m = c2;
        break;
      }
    } else {
      break;
    }
  }
  if (now_c - c_m < min_move_c_)
    c_m = now_c - min_move_c_;
  return c_m;
}

double Validation_error_path::get_apprx_c_bisec_left_cv(
    const std::vector<Eigen::VectorXd> &w_set,
    const std::vector<Eigen::VectorXd> &grad_w_set, const double &tolerance,
    const double &now_c, const double &pre_c, const double &accuracy) {

  double c1 = now_c;
  double c2 = pre_c;
  double c_m, mean_lb_ve_cm;
  double limit_lb = best_valid_error_ - tolerance;
  if (limit_lb <= 0.0)
    limit_lb = DBL_MIN;

  while (1) {
    c_m = 0.5 * (c1 + c2);
    mean_lb_ve_cm = 0.0;
    for (int i = 0; i < fold_num_; ++i)
      mean_lb_ve_cm += (train_objs_[i])->get_lower_bound_valid_error(
          w_set[i], grad_w_set[i], now_c, c_m);
    mean_lb_ve_cm /= fold_num_;
    if (mean_lb_ve_cm > limit_lb) {
      c1 = c_m;
      if (c1 - c2 <= accuracy) {
        c_m = c1;
        break;
      }
    } else if (mean_lb_ve_cm < limit_lb) {
      c2 = c_m;
      if (c1 - c2 <= accuracy) {
        c_m = c2;
        break;
      }
    } else {
      break;
    }
  }
  if (now_c - c_m < min_move_c_)
    c_m = now_c - min_move_c_;
  return c_m;
}

double Validation_error_path::get_apprx_c_right_inexact(
    const Eigen::VectorXd &w, const Eigen::VectorXd &grad_w,
    const double &tolerance, const double &now_c) {
  Eigen::VectorXd w_x = valid_x_ * w;
  Eigen::VectorXd g_x = valid_x_ * grad_w;
  int tolerance_instance = valid_l_ * tolerance;

  std::vector<double> c_vec;
  int num_lb_ve = 0;
  double w_norm = w.norm();
  double g_norm = grad_w.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0 && (wxi - 0.5 * (gxi - gxi_norm)) < 0.0) {
      c_vec.push_back(now_c * (wxi_norm - wxi) /
                      (wxi_norm + wxi + gxi_norm - gxi));
      ++num_lb_ve;
    } else if (valid_y_.coeffRef(i) < 0.0 &&
               (wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
      c_vec.push_back(now_c * (wxi_norm + wxi) /
                      (wxi_norm - wxi + gxi_norm + gxi));
      ++num_lb_ve;
    }
  }
  std::sort(c_vec.begin(), c_vec.end());
  double next_c;
  int th = num_lb_ve - best_num_error_ + tolerance_instance;

  if (c_vec.empty())
    return c_max_;
  if (th < 1) {
    next_c = c_vec.at(0);
  } else if (th <= static_cast<int>(c_vec.size()) - 1) {
    next_c = c_vec.at(th);
  } else {
    next_c = c_vec.at(c_vec.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Validation_error_path::get_apprx_c_right_inexact(
    const Eigen::VectorXd &w, const Eigen::VectorXd &grad_w,
    const double &tolerance, const double &now_c, int &c_set_th) {
  Eigen::VectorXd w_x = valid_x_ * w;
  Eigen::VectorXd g_x = valid_x_ * grad_w;
  int tolerance_instance = valid_l_ * tolerance;

  std::vector<double> c_vec;
  int num_lb_ve = 0;
  double w_norm = w.norm();
  double g_norm = grad_w.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0 && (wxi - 0.5 * (gxi - gxi_norm)) < 0.0) {
      c_vec.push_back(now_c * (wxi_norm - wxi) /
                      (wxi_norm + wxi + gxi_norm - gxi));
      ++num_lb_ve;
    } else if (valid_y_.coeffRef(i) < 0.0 &&
               (wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
      c_vec.push_back(now_c * (wxi_norm + wxi) /
                      (wxi_norm - wxi + gxi_norm + gxi));
      ++num_lb_ve;
    }
  }
  std::sort(c_vec.begin(), c_vec.end());
  double next_c;
  c_set_th = num_lb_ve - best_num_error_ + tolerance_instance;

  if (c_vec.empty())
    return c_max_;
  if (c_set_th < 1) {
    next_c = c_vec.at(0);
  } else if (c_set_th <= static_cast<int>(c_vec.size()) - 1) {
    next_c = c_vec.at(c_set_th);
  } else {
    next_c = c_vec.at(c_vec.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Validation_error_path::get_apprx_c_left_inexact(
    const Eigen::VectorXd &w, const Eigen::VectorXd &grad_w,
    const double &tolerance, const double &now_c) {
  Eigen::VectorXd w_x = valid_x_ * w;
  Eigen::VectorXd g_x = valid_x_ * grad_w;
  int tolerance_instance = valid_l_ * tolerance;

  std::vector<double> c_vec;
  int num_lb_ve = 0;
  double w_norm = w.norm();
  double g_norm = grad_w.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0 && (wxi - 0.5 * (gxi - gxi_norm)) < 0.0) {
      c_vec.push_back(now_c * (wxi_norm + wxi) /
                      (wxi_norm - wxi + gxi_norm + gxi));
      ++num_lb_ve;
    } else if (valid_y_.coeffRef(i) < 0.0 &&
               (wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
      c_vec.push_back(now_c * (wxi_norm - wxi) /
                      (wxi_norm + wxi + gxi_norm - gxi));
      ++num_lb_ve;
    }
  }
  std::sort(c_vec.begin(), c_vec.end(), std::greater<double>());
  double next_c;
  int th = num_lb_ve - best_num_error_ + tolerance_instance;

  if (c_vec.empty())
    return c_min_;
  if (th < 1) {
    next_c = c_vec.at(0);
  } else if (th <= static_cast<int>(c_vec.size()) - 1) {
    next_c = c_vec.at(th);
  } else {
    next_c = c_vec.at(c_vec.size() - 1);
  }

  if (now_c - next_c < min_move_c_)
    next_c = now_c - min_move_c_;

  return next_c;
}

double Validation_error_path::get_apprx_c_right_exact_cv(
    const std::vector<Eigen::VectorXd> &w_set, const double &tolerance,
    const double &now_c) {
  std::vector<double> whole_c, tmp_c_vec;
  for (int i = 0; i < fold_num_; ++i) {
    tmp_c_vec.clear();
    tmp_c_vec = (train_objs_[i])->get_c_set_right_opt(w_set[i], now_c);
    whole_c.insert(whole_c.end(), tmp_c_vec.begin(), tmp_c_vec.end());
  }
  std::sort(whole_c.begin(), whole_c.end());
  double next_c;
  int num_lb_ve = whole_c.size();
  int tolerance_instance = whole_l_ * tolerance;
  int th = num_lb_ve - best_num_error_ + tolerance_instance;

  if (whole_c.empty())
    return c_max_;
  if (th < 1) {
    next_c = whole_c.at(0);
  } else if (th <= static_cast<int>(whole_c.size()) - 1) {
    next_c = whole_c.at(th);
  } else {
    next_c = whole_c.at(whole_c.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Validation_error_path::get_apprx_c_right_inexact_cv(
    const std::vector<Eigen::VectorXd> &w_set,
    const std::vector<Eigen::VectorXd> &grad_w_set, const double &tolerance,
    const double &now_c) {
  std::vector<double> whole_c, tmp_c_vec;
  for (int i = 0; i < fold_num_; ++i) {
    tmp_c_vec.clear();
    tmp_c_vec = (train_objs_[i])
                    ->get_c_set_right_subopt(w_set[i], grad_w_set[i], now_c);
    whole_c.insert(whole_c.end(), tmp_c_vec.begin(), tmp_c_vec.end());
  }
  std::sort(whole_c.begin(), whole_c.end());
  double next_c;
  int num_lb_ve = whole_c.size();
  int tolerance_instance = whole_l_ * tolerance;
  int th = num_lb_ve - best_num_error_ + tolerance_instance;

  if (whole_c.empty())
    return c_max_;
  if (th < 1) {
    next_c = whole_c.at(0);
  } else if (th <= static_cast<int>(whole_c.size()) - 1) {
    next_c = whole_c.at(th);
  } else {
    next_c = whole_c.at(whole_c.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Validation_error_path::get_apprx_c_left_inexact_cv(
    const std::vector<Eigen::VectorXd> &w_set,
    const std::vector<Eigen::VectorXd> &grad_w_set, const double &tolerance,
    const double &now_c) {
  std::vector<double> whole_c, tmp_c_vec;
  for (int i = 0; i < fold_num_; ++i) {
    tmp_c_vec.clear();
    tmp_c_vec =
        (train_objs_[i])->get_c_set_left_subopt(w_set[i], grad_w_set[i], now_c);
    whole_c.insert(whole_c.end(), tmp_c_vec.begin(), tmp_c_vec.end());
  }
  std::sort(whole_c.begin(), whole_c.end(), std::greater<double>());
  double next_c;
  int num_lb_ve = whole_c.size();
  int tolerance_instance = whole_l_ * tolerance;
  int th = num_lb_ve - best_num_error_ + tolerance_instance;

  if (whole_c.empty())
    return c_min_;
  if (th < 1) {
    next_c = whole_c.at(0);
  } else if (th <= static_cast<int>(whole_c.size()) - 1) {
    next_c = whole_c.at(th);
  } else {
    next_c = whole_c.at(whole_c.size() - 1);
  }

  if (now_c - next_c < min_move_c_)
    next_c = now_c - min_move_c_;

  return next_c;
}

double Validation_error_path::get_apprx_c_right_exact_cv_for_path(
    const std::vector<Eigen::VectorXd> &w_set, const double &tolerance,
    const double &now_c) {
  std::vector<double> whole_c, tmp_c_vec;
  for (int i = 0; i < fold_num_; ++i) {
    tmp_c_vec.clear();
    tmp_c_vec = (train_objs_[i])->get_c_set_right_opt_for_path(w_set[i], now_c);
    whole_c.insert(whole_c.end(), tmp_c_vec.begin(), tmp_c_vec.end());
  }
  std::sort(whole_c.begin(), whole_c.end());
  double next_c;
  int th = whole_l_ * tolerance;

  if (whole_c.empty())
    return c_max_;
  if (th < 1) {
    next_c = whole_c.at(0);
  } else if (th <= static_cast<int>(whole_c.size()) - 1) {
    next_c = whole_c.at(th);
  } else {
    next_c = whole_c.at(whole_c.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Validation_error_path::get_apprx_c_right_inexact_cv_for_path(
    const std::vector<Eigen::VectorXd> &w_set,
    const std::vector<Eigen::VectorXd> &grad_w_set, const double &tolerance,
    const double &now_c) {
  std::vector<double> whole_c, tmp_c_vec;
  int total_num_dif = 0, num_dif_ublb = 0;
  for (int i = 0; i < fold_num_; ++i) {
    tmp_c_vec.clear();
    tmp_c_vec = (train_objs_[i])->get_c_set_right_subopt_for_path(
        w_set[i], grad_w_set[i], now_c, num_dif_ublb);
    whole_c.insert(whole_c.end(), tmp_c_vec.begin(), tmp_c_vec.end());
    total_num_dif += num_dif_ublb;
  }
  std::sort(whole_c.begin(), whole_c.end());
  double next_c;
  int th = whole_l_ * tolerance - total_num_dif;
  // std::cout << th <<" " << (double)total_num_dif/whole_l_ <<std::endl;
  if (whole_c.empty())
    return c_max_;
  if (th < 1) {
    next_c = whole_c.at(0);
  } else if (th <= static_cast<int>(whole_c.size()) - 1) {
    next_c = whole_c.at(th);
  } else {
    next_c = whole_c.at(whole_c.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;

  return next_c;
}

double Validation_error_path::get_upper_bound_valid_error(
    const Eigen::VectorXd &w, const Eigen::VectorXd &func_grad) {
  Eigen::VectorXd xm = 0.5 * (valid_x_ * (2.0 * w - func_grad));
  double r = 0.5 * (func_grad).norm();
  double ub_ve = 0.0;
  for (int i = 0; i < valid_l_; ++i) {
    if (valid_y_.coeffRef(i) > 0.0) {
      if (xm.coeffRef(i) - r * valid_x_norm_.coeffRef(i) > 0.0)
        ++ub_ve;
    } else {
      if (xm.coeffRef(i) + r * valid_x_norm_.coeffRef(i) < 0.0)
        ++ub_ve;
    }
  }
  int num_err = valid_l_ - ub_ve;
  if (best_num_error_ > num_err)
    best_num_error_ = num_err;
  ub_ve /= valid_l_;
  return 1.0 - ub_ve;
}

double Validation_error_path::get_lower_bound_valid_error(
    const Eigen::VectorXd &w, const Eigen::VectorXd &c_loss_grad) {
  Eigen::VectorXd xm = 0.5 * (valid_x_ * (w - c_loss_grad));
  double r = 0.5 * (w + c_loss_grad).norm();
  double lb_ve = 0.0;
  for (int i = 0; i < valid_l_; ++i) {
    if (valid_y_.coeffRef(i) > 0.0) {
      if (xm.coeffRef(i) + r * valid_x_norm_.coeffRef(i) < 0.0)
        ++lb_ve;
    } else {
      if (xm.coeffRef(i) - r * valid_x_norm_.coeffRef(i) > 0.0)
        ++lb_ve;
    }
  }
  lb_ve /= valid_l_;
  return lb_ve;
}

void Validation_error_path::recursive_check(const Eigen::VectorXd &w_c1,
                                            const Eigen::VectorXd &w_c2,
                                            const double &c1, const double &c2,
                                            const double &epsilon, int &iter,
                                            int &re_iter) {
  if (c2 - c1 <= min_move_c_ || c2 >= c_max_) {
    return;
  }

  double valid_error;
  double c1_tilde =
      get_approximate_c_only_error(w_c1, c1, epsilon, valid_error);
  double c2_tilde = get_apprx_c_prev(w_c2, c2, epsilon, valid_error);

  if (c1_tilde < c2_tilde) {
    double c_m = 0.5 * (c1_tilde + c2_tilde);

    (train_obj_->get_fun_obj())->set_regularized_parameter(c_m);
    Eigen::VectorXd w_cm = train_obj_->train_warm_start(w_c1);
    double ve_cm = get_valid_error(w_cm);
    ++re_iter;
    ++iter;
    std::cout << "recursive !!!! " << iter << " " << c_m << " " << ve_cm
              << std::endl;
    recursive_check(w_c1, w_cm, c1, c_m, epsilon, iter, re_iter);
    recursive_check(w_cm, w_c2, c_m, c2, epsilon, iter, re_iter);
  } else {
    return;
  }
}

void Validation_error_path::recursive_check_bisec(
    const Eigen::VectorXd &w_c1, const Eigen::VectorXd &w_c2,
    const Eigen::VectorXd &grad_w1, const Eigen::VectorXd &grad_w2,
    const double &c1, const double &c2, const double &epsilon,
    const double &inexact_level, int &iter, int &re_iter) {

  if (c2 - c1 <= min_move_c_ || c2 >= c_max_) {
    return;
  }

  double c1_tilde = get_apprx_c_bisec_right(w_c1, grad_w1, epsilon, c1);
  double c2_tilde = get_apprx_c_bisec_left(w_c2, grad_w2, epsilon, c2, c1);
  double ub_ve_cm, lb_ve_cm;
  Eigen::VectorXd grad_w_cm;
  if (c1_tilde < c2_tilde) {
    double c_m = 0.5 * (c1_tilde + c2_tilde);
    (train_obj_->get_fun_obj())->set_regularized_parameter(c_m);
    Eigen::VectorXd w_cm = train_obj_->train_warm_start_inexact(
        w_c1, inexact_level, ub_ve_cm, lb_ve_cm);
    update_best(ub_ve_cm, c_m);
    ++re_iter;
    ++iter;
    std::cout << "recursive !!!! " << iter << " " << c_m << " " << ub_ve_cm
              << " " << c1_tilde << " " << c2_tilde << std::endl;
    grad_w_cm = train_obj_->get_grad();
    recursive_check_bisec(w_c1, w_cm, grad_w1, grad_w_cm, c1, c_m, epsilon,
                          inexact_level, iter, re_iter);
    recursive_check_bisec(w_cm, w_c2, grad_w_cm, grad_w2, c_m, c2, epsilon,
                          inexact_level, iter, re_iter);
  } else {
    return;
  }
}

void Validation_error_path::recursive_check_bisec_cv(
    const std::vector<Eigen::VectorXd> &w_c1,
    const std::vector<Eigen::VectorXd> &w_c2,
    const std::vector<Eigen::VectorXd> &grad_w1,
    const std::vector<Eigen::VectorXd> &grad_w2, const double &c1,
    const double &c2, const double &epsilon, const double &inexact_level,
    int &iter, int &re_iter) {
  if (c2 - c1 <= min_move_c_ || c2 >= c_max_) {
    return;
  }

  double c1_tilde = get_apprx_c_bisec_right_cv(w_c1, grad_w1, epsilon, c1);
  // std::cout << "pre recursive train1" << std::endl;

  double c2_tilde = get_apprx_c_bisec_left_cv(w_c2, grad_w2, epsilon, c2, c1);
  double ub_ve_cm, lb_ve_cm, mean_ub_ve_cm, mean_lb_ve_cm;
  std::vector<Eigen::VectorXd> w_cm_set, grad_w_cm_set;
  // std::cout << "recursive "
  //           << " " << c1_tilde << " " << c2_tilde << std::endl;
  if (c1_tilde < c2_tilde) {
    double c_m = 0.5 * (c1_tilde + c2_tilde);
    mean_ub_ve_cm = 0.0;
    mean_lb_ve_cm = 0.0;
    for (int i = 0; i < fold_num_; ++i) {
      // std::cout << "pre recursive train" << std::endl;
      (train_objs_[i]->get_fun_obj())->set_regularized_parameter(c_m);
      (train_objs_[i])->train_warm_start_inexact(w_c1[i], inexact_level,
                                                 ub_ve_cm, lb_ve_cm);
      w_cm_set.push_back(train_objs_[i]->get_w());
      grad_w_cm_set.push_back(train_objs_[i]->get_grad());
      mean_ub_ve_cm += ub_ve_cm;
      mean_lb_ve_cm += lb_ve_cm;
    }
    mean_ub_ve_cm /= fold_num_;
    mean_lb_ve_cm /= fold_num_;

    update_best(mean_ub_ve_cm, c_m);
    ++re_iter;
    ++iter;
    std::cout << "recursive !!!! " << iter << " " << c_m << " " << mean_ub_ve_cm
              << " " << c1_tilde << " " << c2_tilde << std::endl;
    recursive_check_bisec_cv(w_c1, w_cm_set, grad_w1, grad_w_cm_set, c1, c_m,
                             epsilon, inexact_level, iter, re_iter);
    recursive_check_bisec_cv(w_cm_set, w_c2, grad_w_cm_set, grad_w2, c_m, c2,
                             epsilon, inexact_level, iter, re_iter);
  } else {
    return;
  }
}

void Validation_error_path::recursive_check_inexact(
    const Eigen::VectorXd &w_c1, const Eigen::VectorXd &w_c2,
    const Eigen::VectorXd &grad_w1, const Eigen::VectorXd &grad_w2,
    const double &c1, const double &c2, const double &epsilon,
    const double &inexact_level, int &iter, int &re_iter) {

  if (c2 - c1 <= min_move_c_ || c2 >= c_max_) {
    return;
  }

  double c1_tilde = get_apprx_c_right_inexact(w_c1, grad_w1, epsilon, c1);
  double c2_tilde = get_apprx_c_left_inexact(w_c2, grad_w2, epsilon, c2);
  double ub_ve_cm, lb_ve_cm;
  Eigen::VectorXd grad_w_cm;
  if (c1_tilde < c2_tilde) {
    double c_m = 0.5 * (c1_tilde + c2_tilde);
    (train_obj_->get_fun_obj())->set_regularized_parameter(c_m);
    Eigen::VectorXd w_cm = train_obj_->train_warm_start_inexact(
        w_c1, inexact_level, ub_ve_cm, lb_ve_cm);
    int ttt = ub_ve_cm * valid_l_;
    update_best(ttt, ub_ve_cm, c_m, w_cm);
    ++re_iter;
    ++iter;
    std::cout << iter << " " << c_m << " " << ub_ve_cm << " " << c1_tilde << " "
              << c2_tilde << " " << 0 << std::endl;
    grad_w_cm = train_obj_->get_grad();
    recursive_check_inexact(w_c1, w_cm, grad_w1, grad_w_cm, c1, c_m, epsilon,
                            inexact_level, iter, re_iter);
    recursive_check_inexact(w_cm, w_c2, grad_w_cm, grad_w2, c_m, c2, epsilon,
                            inexact_level, iter, re_iter);
  } else {
    return;
  }
}

void Validation_error_path::recursive_check_inexact_cv(
    const std::vector<Eigen::VectorXd> &w_c1,
    const std::vector<Eigen::VectorXd> &w_c2,
    const std::vector<Eigen::VectorXd> &grad_w1,
    const std::vector<Eigen::VectorXd> &grad_w2, const double &c1,
    const double &c2, const double &epsilon, const double &inexact_level,
    int &iter, int &re_iter) {
  if (c2 - c1 <= min_move_c_ || c2 >= c_max_) {
    return;
  }

  double c1_tilde = get_apprx_c_right_inexact_cv(w_c1, grad_w1, epsilon, c1);
  double c2_tilde = get_apprx_c_left_inexact_cv(w_c2, grad_w2, epsilon, c2);
  double ub_ve_cm, lb_ve_cm, mean_ub_ve_cm;
  std::vector<Eigen::VectorXd> w_cm_set, grad_w_cm_set;
  int ttt = 0, tt;
  if (c1_tilde < c2_tilde) {
    double c_m = 0.5 * (c1_tilde + c2_tilde);
    mean_ub_ve_cm = 0.0;
    for (int i = 0; i < fold_num_; ++i) {
      (train_objs_[i]->get_fun_obj())->set_regularized_parameter(c_m);
      (train_objs_[i])->train_warm_start_inexact(w_c1[i], inexact_level,
                                                 ub_ve_cm, lb_ve_cm);
      w_cm_set.push_back(train_objs_[i]->get_w());
      grad_w_cm_set.push_back(train_objs_[i]->get_grad());
      tt = ub_ve_cm * (train_objs_[i])->get_valid_l();
      ttt += tt;
    }
    mean_ub_ve_cm = (double)ttt / whole_l_;

    update_best(ttt, mean_ub_ve_cm, c_m);
    ++re_iter;
    ++iter;
    std::cout <<"recursive : " << iter << " " << c_m << " " << mean_ub_ve_cm << " " << c1_tilde
              << " " << c2_tilde << " 0" << std::endl;
    recursive_check_inexact_cv(w_c1, w_cm_set, grad_w1, grad_w_cm_set, c1, c_m,
                               epsilon, inexact_level, iter, re_iter);
    recursive_check_inexact_cv(w_cm_set, w_c2, grad_w_cm_set, grad_w2, c_m, c2,
                               epsilon, inexact_level, iter, re_iter);
  } else {
    return;
  }
}

double Validation_error_path::get_apprx_path_c_inexact(
    const Eigen::VectorXd &w, const Eigen::VectorXd &grad_w,
    const double &tolerance, const double &now_c) {
  Eigen::VectorXd w_x = valid_x_ * w;
  Eigen::VectorXd g_x = valid_x_ * grad_w;
  int tolerance_instance = valid_l_ * tolerance;

  std::vector<double> c_vec;
  int num_lb_ve = 0, num_ub_ve = 0;
  double w_norm = w.norm();
  double g_norm = grad_w.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0) {
      if ((wxi - 0.5 * (gxi - gxi_norm)) < 0.0) {
        c_vec.push_back(now_c * (wxi_norm - wxi) /
                        (wxi_norm + wxi + gxi_norm - gxi));
        ++num_lb_ve;
      } else if ((wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
        c_vec.push_back(now_c * (wxi_norm + wxi) /
                        (wxi_norm - wxi + gxi_norm + gxi));
        ++num_ub_ve;
      }
    } else if (valid_y_.coeffRef(i) < 0.0) {
      if ((wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
        c_vec.push_back(now_c * (wxi_norm + wxi) /
                        (wxi_norm - wxi + gxi_norm + gxi));
        ++num_lb_ve;
      } else if ((wxi - 0.5 * (gxi - gxi_norm)) < 0.0) {
        c_vec.push_back(now_c * (wxi_norm - wxi) /
                        (wxi_norm + wxi + gxi_norm - gxi));
        ++num_ub_ve;
      }
    }
  }
  num_ub_ve = valid_l_ - num_ub_ve;
  std::sort(c_vec.begin(), c_vec.end());
  double next_c;
  int th = tolerance_instance - num_ub_ve + num_lb_ve;
  // std::cout << th << std::endl;

  if (th < 1) {
    next_c = c_vec.at(0);
  } else if (th <= static_cast<int>(c_vec.size()) - 1) {
    next_c = c_vec.at(th);
  } else {
    next_c = c_vec.at(c_vec.size() - 1);
  }

  if (next_c - now_c < min_move_c_)
    next_c = now_c + min_move_c_;
  // std::cout << valid_error << ", "  << tmp_num_err <<", " << valid_l_
  // <<std::endl;
  return next_c;
}
