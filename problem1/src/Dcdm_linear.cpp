#include "Dcdm_linear.h"

namespace sdm {

Dcdm_linear::Dcdm_linear(const std::string &train, const double &c,
                         const double &stpctr, const int &max_iter)
    : train_x_(), train_y_(), train_l_(), train_n_(), kernel_(), C(c),
      stopping_criterion(stpctr), max_iteration(max_iter), valid_x_(),
      valid_y_(), valid_l_(0), valid_n_(0), w_(), grad_(), wx_(), x_sqnorm_() {
  sdm::load_libsvm_binary(train_x_, train_y_, train);
  train_l_ = train_x_.rows();
  train_n_ = train_x_.cols();
  set_Q();
  x_sqnorm_.resize(train_l_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < train_l_; ++i) {
    x_sqnorm_[i] = (train_x_.row(i)).squaredNorm();
  }

  w_.resize(train_n_);
  grad_ = -Eigen::VectorXd::Ones(train_l_);
  wx_.resize(train_l_);
}

Dcdm_linear::Dcdm_linear(const std::string &train, const std::string &valid,
                         const double &c, const double &stpctr,
                         const int &max_iter)
    : train_x_(), train_y_(), train_l_(), train_n_(), kernel_(), C(c),
      stopping_criterion(stpctr), max_iteration(max_iter), valid_x_(),
      valid_y_(), valid_l_(), valid_n_(), w_(), grad_(), wx_(), x_sqnorm_() {
  sdm::load_libsvm_binary(train_x_, train_y_, train);
  train_l_ = train_x_.rows();
  train_n_ = train_x_.cols();
  set_Q();
  x_sqnorm_.resize(train_l_);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < train_l_; ++i) {
    x_sqnorm_[i] = (train_x_.row(i)).squaredNorm();
  }

  sdm::load_libsvm_binary(valid_x_, valid_y_, valid);
  valid_l_ = valid_x_.rows();
  valid_n_ = valid_x_.cols();

  w_.resize(train_n_);
  grad_ = -Eigen::VectorXd::Ones(train_l_);
  wx_.resize(train_l_);
}

Dcdm_linear::~Dcdm_linear() {}

void Dcdm_linear::set_regularized_parameter(const double &c) { C = c; }

int Dcdm_linear::get_train_l(void) { return train_l_; }
int Dcdm_linear::get_valid_l(void) { return valid_l_; }
int Dcdm_linear::get_valid_n(void) { return valid_n_; }

void Dcdm_linear::set_Q(void) {
  kernel_.resize(train_l_, train_l_);
  double kij = 0.0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < train_l_; ++i) {
    for (int j = i; j < train_l_; ++j) {
      kij = train_y_[i] * train_y_[j] * train_x_.row(i).dot(train_x_.row(j));
      kernel_.coeffRef(i, j) = kij;
      kernel_.coeffRef(j, i) = kij;
    }
  }
}

void Dcdm_linear::update_grad(const double &delta, const unsigned int &index) {
  grad_ += delta * kernel_.row(index).transpose();
}

double Dcdm_linear::predict(const Eigen::VectorXd &w) const {

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

Eigen::VectorXd Dcdm_linear::train_warm_start(const Eigen::VectorXd &alp) {
  Eigen::VectorXd alpha = Eigen::VectorXd::Zero(train_l_); // alp;
  w_ = Eigen::VectorXd::Zero(train_n_);
  std::cout << "ok0" << std::endl;
  double alpha_old = 0.0;
  // main loop
  for (int iter = 1; iter <= max_iteration; ++iter) {
    // unsigned int max_idx = 0;
    // double max_violation = 0.0;
    double PG = 0.0; // PG: projected gradient

    for (int i = 0; i < train_l_; ++i) {
      grad_[i] = (train_y_[i] * (train_x_.row(i) * w_)(0)) - 1.0;
      if (alpha[i] == (double)0.0) {
        PG = std::min(grad_[i], 0.0);
      } else if (alpha[i] == C) {
        PG = std::max(grad_[i], 0.0);
      } else {
        PG = grad_[i];
      }

      if (PG != 0.0) {
        alpha_old = alpha[i];
        alpha[i] =
            std::min(std::max(alpha[i] - grad_[i] / x_sqnorm_[i], 0.0), C);
        w_ +=
            train_y_[i] * (alpha[i] - alpha_old) * train_x_.row(i).transpose();
      }
    }
    double grad_norm = 1.0;
    if (iter % 2 == 0) {
      Eigen::VectorXd tmp_vd = Eigen::VectorXd::Zero(train_n_);
      for (int i = 0; i < train_l_; ++i) {
        tmp_vd += train_y_[i] * alpha[i] * train_x_.row(i).transpose();
      }
      double w_norm = tmp_vd.norm();
      grad_norm = get_grad_norm(w_);
      std::cout << "iter: " << iter // << ", max_violation: " << max_violation
                << ", f: " << get_primal_func(w_)
                << ", d: " << get_dual_func(alpha) << ", ||g|| : " << grad_norm
                << ", ||w|| : " << w_.norm() << ", " << w_norm << std::endl;
    }

    if (grad_norm <= stopping_criterion)
      break;
  }
  return w_;
}

double Dcdm_linear::get_dual_func(const Eigen::VectorXd &alpha) {
  return 0.5 * (alpha.transpose() * (kernel_ * alpha))(0) - alpha.sum();
}

double Dcdm_linear::get_primal_func(const Eigen::VectorXd &w) {

  Eigen::ArrayXd wx = train_y_ * (train_x_ * w).array();
  double tmp = 0.0;
  for (int i = 0; i < train_l_; ++i) {
    if (wx[i] < 1.0)
      tmp += 1.0 - wx[i];
  }
  return 0.5 * w.squaredNorm() + C * tmp;
}

Eigen::VectorXd Dcdm_linear::get_grad(const Eigen::VectorXd &w) {
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

double Dcdm_linear::get_grad_norm(const Eigen::VectorXd &w) {
  return ((get_grad(w)).norm());
}

std::vector<double>
Dcdm_linear::get_c_set_right_opt(const Eigen::VectorXd &w_opt,
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
