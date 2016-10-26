#include "Dcdm_l2svm.h"

namespace sdm {

Dcdm_l2svm::Dcdm_l2svm(const std::string &train, const double &c,
                       const double &gmm, const double &stpctr,
                       const unsigned int &max_iter)
    : train_x_(), train_y_(), train_l_(), train_n_(), kernel_(), C(c), C05(),
      gamma(gmm), stopping_criterion(stpctr), max_iteration(max_iter),
      valid_x_(), valid_y_(), valid_l_(0), valid_n_(0), grad_(), wx_(), z_() {
  sdm::load_libsvm_binary(train_x_, train_y_, train);
  train_l_ = train_x_.rows();
  train_n_ = train_x_.cols();
  set_rbf_kernel();

  grad_ = -Eigen::VectorXd::Ones(train_l_);
  C05 = 1.0 / (2.0 * C);
  wx_.resize(train_l_);
  z_.resize(train_l_);
}

Dcdm_l2svm::Dcdm_l2svm(const std::string &train, const std::string &valid,
                       const double &c, const double &gmm, const double &stpctr,
                       const unsigned int &max_iter)
    : train_x_(), train_y_(), train_l_(), train_n_(), kernel_(), C(c), C05(),
      gamma(gmm), stopping_criterion(stpctr), max_iteration(max_iter),
      valid_x_(), valid_y_(), valid_l_(), valid_n_(), grad_(), wx_(), z_() {
  sdm::load_libsvm_binary(train_x_, train_y_, train);
  train_l_ = train_x_.rows();
  train_n_ = train_x_.cols();
  set_rbf_kernel();

  sdm::load_libsvm_binary(valid_x_, valid_y_, valid);
  valid_l_ = valid_x_.rows();
  valid_n_ = valid_x_.cols();
  set_rbf_kernel_valid();

  grad_ = -Eigen::VectorXd::Ones(train_l_);
  C05 = 1 / (2.0 * C);
  wx_.resize(train_l_);
  z_.resize(train_l_);
}

Dcdm_l2svm::~Dcdm_l2svm() {}

void Dcdm_l2svm::set_regularized_parameter(const double &c) { C = c; }
void Dcdm_l2svm::set_kernel_parameter(const double &gmm) { gamma = gmm; }

int Dcdm_l2svm::get_train_l(void) { return train_l_; }
int Dcdm_l2svm::get_valid_l(void) { return valid_l_; }

void Dcdm_l2svm::set_rbf_kernel(void) {
  kernel_.resize(train_l_, train_l_);
// double kij = 0.0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < train_l_; ++i) {
    kernel_.coeffRef(i, i) = 1.0;
    for (int j = i+1; j < train_l_; ++j) {
      double kij =
          train_y_[i] * train_y_[j] *
          (fmath::exp(-gamma *
                      (train_x_.row(i) - train_x_.row(j)).squaredNorm()));
      kernel_.coeffRef(i, j) = kij;
      kernel_.coeffRef(j, i) = kij;
    }
  }
}

void Dcdm_l2svm::set_rbf_kernel_valid(void) {
  kernel_valid_.resize(valid_l_, train_l_);
  double kij = 0.0;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < valid_l_; ++i) {
    for (int j = 0; j < train_l_; ++j) {
      if (i == j) {
        kij = 1.0;
      } else {
        kij = fmath::exp(-gamma *
                         (valid_x_.row(i) - train_x_.row(j)).squaredNorm());
      }
      kernel_valid_.coeffRef(i, j) = kij;
    }
  }
}

double Dcdm_l2svm::predict(const Eigen::ArrayXd &alpha) const {
  int miss = 0;
  if (valid_l_ > 0) {
    // std::cout << alpha.size() <<" " << train_y_.size() <<" " <<
    // kernel_valid_.cols() << std::endl;

    // std::cout << (alpha.array() *
    // train_y_).matrix().dot(kernel_valid_.row(0)) << std::endl;

    for (int i = 0; i < valid_l_; ++i) {
      double fx = (alpha * train_y_).matrix().dot(kernel_valid_.row(i));
      if (fx * valid_y_.coeffRef(i) < 0.0) {
        ++miss;
        // std::cout << "false " << valid_y_.coeffRef(i) << " " << fx <<
        // std::endl;
      } // else {
        //   // std::cout << "true " << valid_y_.coeffRef(i) << " " << fx <<
        //   std::endl;
        // }
    }
  }
  // std::cout << alpha.transpose() <<std::endl;
  return (static_cast<double>(miss)) / valid_l_;
}

void Dcdm_l2svm::update_grad(const double &delta, const unsigned int &index) {
  grad_ += delta * kernel_.row(index).transpose();
}

Eigen::ArrayXd Dcdm_l2svm::train_warm_start(const Eigen::ArrayXd &alp) {
  Eigen::ArrayXd alpha = Eigen::VectorXd::Zero(train_l_); // alp;

  // main loop
  for (unsigned int iter = 1; iter <= max_iteration; ++iter) {
    // unsigned int max_idx = 0;
    // double max_violation = 0.0;
    double PG = 0.0; // PG: projected gradient

    for (int i = 0; i < train_l_; ++i) {
      grad_[i] = kernel_.row(i).dot(alpha.matrix()) - 1.0 + alpha[i] * C05;
      if (alpha[i] == (double)0.0) {
        PG = std::min(grad_[i], 0.0);
      } else {
        PG = grad_[i];
      }

      //   if (std::abs(PG) > max_violation) {
      //     max_violation = std::abs(PG);
      //     max_idx = i;
      //   }
      // }

      if (PG != 0.0)
        alpha[i] = std::max(alpha[i] - grad_[i] / (1.0 + C05), (double)0.0);
    }

    double grad_norm = 1.0;
    if (iter % 2 == 0) {
      grad_norm = get_grad_norm(alpha);
      // std::cout << "iter: " << iter // << ", max_violation: " << max_violation
      //           << ", f: " << get_primal_func(alpha)
      //           << ", d: " << get_dual_func(alpha) << ", ||g|| : " << grad_norm
      //           << std::endl;
    }

    if (grad_norm <= stopping_criterion)
      break;

    // if (max_violation < stopping_criterion)
    //   break;

    // double alpha_old = alpha[max_idx];
    // alpha[max_idx] =
    //     std::max(alpha[max_idx] - grad_[max_idx] / (1.0 + C05), (double)0.0);
    // update_grad(alpha[max_idx] - alpha_old, max_idx);
  }
  return alpha;
}

double Dcdm_l2svm::get_dual_func(const Eigen::ArrayXd &alpha) {
  Eigen::VectorXd alp = alpha.matrix();
  return 0.5 * ((alp.transpose() * (kernel_ * alp)) + alp.dot(C05 * alp)) -
         alpha.sum();
}

double Dcdm_l2svm::get_primal_func(const Eigen::ArrayXd &alpha) {
  set_wx_z(alpha);
  double tmp = 0.0;
  for (int i = 0; i < train_l_; ++i) {
    if (z_[i] > 0.0)
      tmp += z_[i] * z_[i];
  }
  // std::cout <<" tmp : " << tmp <<std::endl;
  return 0.5 * get_w_squared_norm(alpha) + C * tmp;
}

double Dcdm_l2svm::get_w_squared_norm(const Eigen::ArrayXd &alpha) {
  Eigen::VectorXd alp = alpha.matrix();
  return (alp.transpose() * (kernel_ * alp));
}

void Dcdm_l2svm::set_wx_z(const Eigen::ArrayXd &alpha) {
  Eigen::VectorXd alp = alpha.matrix();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < train_l_; ++i) {
    wx_[i] = train_y_[i] * ((kernel_.row(i)).dot(alp));
    z_[i] = 1.0 - train_y_[i] * wx_[i];
  }
}

double Dcdm_l2svm::get_grad_norm(const Eigen::ArrayXd &alpha) {
  double w_partial_l = 0.0;
  double partial_l_squared_norm = 0.0;
  double tmp, tmp1;
  set_wx_z(alpha);
  for (int i = 0; i < train_l_; ++i) {
    tmp = z_[i];
    if (tmp > 0.0) {
      w_partial_l += 4.0 * tmp * (tmp - 1.0);
      partial_l_squared_norm += 4.0 * tmp * tmp;
      for (int j = i + 1; j < train_l_; ++j) {
        tmp1 = z_[j];
        if (tmp1 > 0.0)
          partial_l_squared_norm += 8.0 * tmp * tmp1 * kernel_.coeffRef(i, j);
      }
    }
  }
  tmp = sqrt(get_w_squared_norm(alpha) + C * w_partial_l +
             C * C * partial_l_squared_norm);
  if (std::isnan(tmp))
    tmp = 1e-6;
  return tmp;
}

std::vector<double>
Dcdm_l2svm::get_c_set_right_opt(const Eigen::ArrayXd &alpha_opt,
                                const double &c_now, double &valid_err) {

  Eigen::VectorXd alp = alpha_opt.matrix();
  double w_norm = sqrt(get_w_squared_norm(alpha_opt));
  // Eigen::ArrayXd w_xprime = valid_y_ * (kernel_valid_ * alp).array();

  std::vector<double> c_vec;
  int tmp_num_err = 0;
  double zi = 0.0;
  for (int i = 0; i < valid_l_; ++i) {
    zi = (alpha_opt * train_y_).matrix().dot(kernel_valid_.row(i));
    if (valid_y_[i] * zi < 0.0) {
      ++tmp_num_err;
      if (zi < 0.0) {
        c_vec.push_back(c_now * (w_norm - zi) / (w_norm + zi));
      } else if (zi > 0.0) {
        c_vec.push_back(c_now * (w_norm + zi) / (w_norm - zi));
      }
    }
  }

  std::cout << "cvec size : " <<c_vec.size() <<" , " << tmp_num_err <<std::endl;
  valid_err = (double)tmp_num_err/valid_l_;
  std::sort(c_vec.begin(), c_vec.end());
  return c_vec;
}

} // namespace sdm
