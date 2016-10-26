#include "Subgrad_method.h"

namespace sdm {

Subgrad_method::Subgrad_method(const std::string &train, const double &c,
                               const double &stpctr,
                               const unsigned int &max_iter)
    : train_x_(), train_y_(), train_l_(), train_n_(), C(c),
      stopping_criterion(stpctr), max_iteration(max_iter), valid_x_(),
      valid_y_(), valid_l_(0), valid_n_(0) {
  sdm::load_libsvm_binary(train_x_, train_y_, train);
  train_l_ = train_x_.rows();
  train_n_ = train_x_.cols();
}

Subgrad_method::Subgrad_method(const std::string &train,
                               const std::string &valid, const double &c,
                               const double &stpctr,
                               const unsigned int &max_iter)
    : train_x_(), train_y_(), train_l_(), train_n_(), C(c),
      stopping_criterion(stpctr), max_iteration(max_iter), valid_x_(),
      valid_y_(), valid_l_(), valid_n_() {
  sdm::load_libsvm_binary(train_x_, train_y_, train);
  train_l_ = train_x_.rows();
  train_n_ = train_x_.cols();

  sdm::load_libsvm_binary(valid_x_, valid_y_, valid);
  valid_l_ = valid_x_.rows();
  valid_n_ = valid_x_.cols();
}

Subgrad_method::~Subgrad_method() {}

void Subgrad_method::set_regularized_parameter(const double &c) { C = c; }

int Subgrad_method::get_train_l(void) { return train_l_; }
int Subgrad_method::get_valid_l(void) { return valid_l_; }
int Subgrad_method::get_valid_n(void) { return valid_n_; }

double Subgrad_method::predict(const Eigen::VectorXd &w) const {

  int miss = 0;
  if (valid_l_ > 0) {
    Eigen::VectorXd w_tmp = w;
    if (train_n_ != valid_n_)
      w_tmp.conservativeResize(valid_n_);
    Eigen::ArrayXd ywx = valid_y_ * (valid_x_ * w_tmp).array();
    for (int i = 0; i < valid_l_; ++i)
      if (ywx[i] < 0.0)
        ++miss;
  }
  return (static_cast<double>(miss)) / valid_l_;
}

Eigen::VectorXd Subgrad_method::train_warm_start(const Eigen::VectorXd &w_) {
  Eigen::VectorXd w = Eigen::VectorXd::Ones(train_n_);
  w *= 0;
  Eigen::VectorXd subgrad, w_new;
  double grad_norm = 1.0;
  // main loop
  double alpha = 0.25, beta = 0.5, step_size = 1.0;
  double f_new, f;
  w_new.resize(train_n_);
  for (unsigned int  iter = 1; iter <= max_iteration; ++iter) {
    f = get_primal_func(w);
    subgrad = get_grad(w);
    w_new.noalias() = w - subgrad;
    f_new = get_primal_func(w_new);
    step_size = 1.0;
    while (f_new > f - alpha * step_size * subgrad.squaredNorm()) {
      if (step_size < 1e-12) {
        std::cout << "WARNING: step size is too small ( step_size < 1e-12) "
                  << std::endl;
        break;
      }
      step_size *= beta;
      w_new.noalias() = w - step_size * subgrad;
      f_new = get_primal_func(w_new);
    }
    w.noalias() = w_new;
    // w -= (C/sqrt(iter)) * subgrad;
    if (iter % 2 == 0) {
      grad_norm = subgrad.norm();
      std::cout << "iter: " << iter // << ", max_violation: " << max_violation
                << ", f: " << get_primal_func(w) << ", ||g|| : " << grad_norm
                << std::endl;
    }

    if (grad_norm <= stopping_criterion)
      break;
  }
  return w;
}

double Subgrad_method::get_primal_func(const Eigen::VectorXd &w) {

  Eigen::ArrayXd wx = train_y_ * (train_x_ * w).array();
  double tmp = 0.0;
  for (int i = 0; i < train_l_; ++i) {
    if (wx[i] < 1.0)
      tmp += 1.0 - wx[i];
  }
  return 0.5 * w.squaredNorm() + C * tmp;
}

Eigen::VectorXd Subgrad_method::get_grad(const Eigen::VectorXd &w) {
  Eigen::ArrayXd wx = train_y_ * (train_x_ * w).array();
  Eigen::VectorXd loss_grad = Eigen::VectorXd::Zero(train_n_);
  for (int i = 0; i < train_l_; ++i) {
    if (wx[i] < 1.0) {
      loss_grad -= train_y_[i] * train_x_.row(i).transpose();

    } else if (wx[i] == 0.0) {
      std::cout << "  !!!! subgradient !!!! " << std::endl;
    }
  }
  return w + C * loss_grad;
}

double Subgrad_method::get_grad_norm(const Eigen::VectorXd &w) {
  return ((get_grad(w)).norm());
}

std::vector<double>
Subgrad_method::get_c_set_right_opt(const Eigen::VectorXd &w_opt,
                                    const double &c_now, double &valid_err) {

  double w_norm = w_opt.norm();
  // Eigen::ArrayXd w_xprime = valid_y_ * (kernel_valid_ * alp).array();

  std::vector<double> c_vec;
  int tmp_num_err = 0;
  Eigen::VectorXd w = w_opt;
  if (w_opt.size() != valid_n_)
    w.conservativeResize(valid_n_);
  Eigen::VectorXd wx = valid_x_ * w;
  double zi;
  for (int i = 0; i < valid_l_; ++i) {
    zi = wx[i];
    if (valid_y_[i] * zi < 0.0) {
      ++tmp_num_err;
      if (zi < 0.0) {
        c_vec.push_back(c_now * (w_norm - zi) / (w_norm + zi));
      } else if (zi > 0.0) {
        c_vec.push_back(c_now * (w_norm + zi) / (w_norm - zi));
      }
    }
  }

  std::cout << "cvec size : " << c_vec.size() << " , " << tmp_num_err
            << std::endl;
  valid_err = (double)tmp_num_err / valid_l_;
  std::sort(c_vec.begin(), c_vec.end());
  return c_vec;
}

} // namespace sdm
