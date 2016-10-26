#include "Tron.h"

namespace sdm {
Tron::Tron(Primal_function *fun_obj, const double &stpctr, const int &max_iter)
    : fun_obj_(fun_obj), stopping_criterion(stpctr), max_iteration(max_iter),
      valid_x_(), valid_y_(), valid_l_(0), valid_n_(0), xm_(), valid_x_norm_(),
      w_(), grad_(), grad_norm_() {}

Tron::Tron(Primal_function *fun_obj, const std::string &valid_libsvm_format,
           const double &stpctr, const int &max_iter)
    : fun_obj_(fun_obj), stopping_criterion(stpctr), max_iteration(max_iter),
      valid_x_(), valid_y_(), valid_l_(), valid_n_(), xm_(), valid_x_norm_(),
      w_(), grad_(), grad_norm_() {
  sdm::load_libsvm_binary(valid_x_, valid_y_, valid_libsvm_format);
  valid_l_ = valid_x_.rows();
  valid_n_ = valid_x_.cols();
  w_ = Eigen::VectorXd::Zero(valid_n_);
  valid_x_norm_.resize(valid_l_);

  // #ifdef _OPENMP
  // #pragma omp parallel for
  // #endif
  for (int i = 0; i < valid_l_; ++i)
    valid_x_norm_[i] = (valid_x_.row(i)).norm();
}

Tron::Tron(Primal_function *fun_obj,
           Eigen::SparseMatrix<double, 1, std::ptrdiff_t> valid_x,
           Eigen::ArrayXd valid_y, const double &stpctr, const int &max_iter)
    : fun_obj_(fun_obj), stopping_criterion(stpctr), max_iteration(max_iter),
      valid_x_(valid_x), valid_y_(valid_y), valid_l_(), valid_n_(), xm_(),
      valid_x_norm_(), w_(), grad_(), grad_norm_() {
  valid_l_ = valid_x_.rows();
  valid_n_ = valid_x_.cols();
  w_ = Eigen::VectorXd::Zero(valid_n_);
  valid_x_norm_.resize(valid_l_);

  // #ifdef _OPENMP
  // #pragma omp parallel for
  // #endif
  for (int i = 0; i < valid_l_; ++i)
    valid_x_norm_.coeffRef(i) = (valid_x_.row(i)).norm();
}

Tron::~Tron() {}

void Tron::set_regularized_parameter(const double &c) {
  fun_obj_->set_regularized_parameter(c);
}

int Tron::get_max_iteration() { return max_iteration; }
double Tron::get_stopping_criterion() { return stopping_criterion; }

// int get_train_l() {return fun_obj_->get_;}
int Tron::get_train_n() { return fun_obj_->get_variable(); }
int Tron::get_valid_l() { return valid_l_; }
int Tron::get_valid_n() { return valid_n_; }

Eigen::VectorXd Tron::get_w() { return w_; }
Eigen::VectorXd Tron::get_grad() { return grad_; }
double Tron::get_grad_norm() { return grad_norm_; }

Eigen::VectorXd Tron::tron() {
  // Parameters for updating the outer_iteratates.
  double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

  // Parameters for updating the trust region size delta.
  double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4.0;

  int n = fun_obj_->get_variable();
  int outer_iteration = 1, cg_iteration = 0, count_rejecting_step = 0;
  double delta, snorm, f, f_new, actual_reduction, model_reduction, gts,
      gnorm_old, gnorm, alpha;

  Eigen::VectorXd w = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd g, r, s, w_new;

  f = fun_obj_->get_func(w);
  g.noalias() = fun_obj_->get_grad(w);

  delta = g.norm();
  gnorm_old = delta;
  gnorm = gnorm_old;

  auto start = std::chrono::system_clock::now();
  auto end = std::chrono::system_clock::now();
  auto diff = end - start;
  std::cout << "iter   time[sec]  ||grad||   f - f_new  CG_iter    ||s||       "
               "delta    num of rejecting s" << std::endl;
  std::cout << "-----+----------+-----------+----------+-------+-----------+---"
               "--------+------------------" << std::endl;
  std::cout << "    0"
            << "          0 " << std::scientific << std::setw(10)
            << std::setprecision(5) << gnorm << " " << std::fixed << std::endl;

  while (outer_iteration < max_iteration) {
    s = tron_cg(delta, g, r, cg_iteration, gnorm);
    w_new.noalias() = w + s;
    f_new = fun_obj_->get_func(w_new);
    actual_reduction = f - f_new;
    gts = g.transpose() * s;
    model_reduction = -0.5 * (gts - (s.transpose() * r));
    snorm = s.norm();

    if (outer_iteration == 1)
      delta = std::min(delta, snorm);
    alpha = 0;
    if (f_new - f - gts <= 0)
      alpha = sigma3;
    else
      alpha = std::max(sigma1, -0.5 * (gts / (f_new - f - gts)));

    // Update the trust region bound according to the ratio of actual to
    // predicted reduction.
    if (actual_reduction < eta0 * model_reduction)
      delta = std::min(std::max(alpha, sigma1) * snorm, sigma2 * delta);
    else if (actual_reduction < eta1 * model_reduction)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma2 * delta));
    else if (actual_reduction < eta2 * model_reduction)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma3 * delta));
    else
      delta = std::max(delta, std::min(alpha * snorm, sigma3 * delta));

    if (actual_reduction > eta0 * model_reduction) {
      w.noalias() = w_new;
      g = fun_obj_->get_grad(w);
      gnorm = g.norm();

      end = std::chrono::system_clock::now();
      diff = end - start;
      std::cout << std::setw(5) << outer_iteration << " " << std::setw(10)
                << std::setprecision(2)
                << double(std::chrono::duration_cast<std::chrono::milliseconds>(
                              diff).count()) /
                       1000 << " " << std::scientific << std::setw(10)
                << std::setprecision(5) << gnorm << " " << (f - f_new) << " "
                << std::fixed << std::setw(6) << cg_iteration << " "
                << std::scientific << std::setw(10) << std::setprecision(5)
                << snorm << " " << delta << std::fixed << " "
                << count_rejecting_step << std::endl;

      count_rejecting_step = 0;
      if (fabs(gnorm - gnorm_old) <= stopping_criterion) {
        break;
      }
      f = f_new;
      gnorm_old = gnorm;
      ++outer_iteration;
    } else {
      // std::cout <<"step s^k was rejected , delta :"  << delta <<", rho :"  <<
      // actual_reduction <<" " <<model_reduction <<std::endl;
      ++count_rejecting_step;
    }
    if (std::isnan(f_new)) {
      std::cout << "WARNING: f_new is not a number\n";
      break;
    }
    if (f < -1.0e+32) {
      std::cout << "WARNING: f < -1.0e+32\n";
      break;
    }
    if (fabs(actual_reduction) <= 0 && model_reduction <= 0) {
      std::cout << "WARNING: actual_reduction and model_reduction <= 0\n";
      break;
    }
    if (fabs(actual_reduction) <= 1.0e-16 * fabs(f) &&
        fabs(model_reduction) <= 1.0e-16 * fabs(f)) {
      std::cout << "WARNING: actual_reduction and model_reduction too small\n";
      std::cout << "actred = " << actual_reduction
                << ", prered = " << model_reduction << std::endl;
      break;
    }
    if (std::isinf(delta)) {
      std::cout << "WARNING: delta is inf \n";
      break;
    }
  }
  return w;
}

Eigen::VectorXd Tron::train_warm_start(const Eigen::VectorXd &w_start) {
  double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
  double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4.0;

  int outer_iteration = 1, cg_iteration = 0, count_rejecting_step = 0;
  double delta, snorm, f, f_new, actual_reduction, model_reduction, gts,
      gnorm_old, alpha;

  w_ = w_start;
  if (w_.size() != fun_obj_->get_variable()) {
    w_.conservativeResize(fun_obj_->get_variable());
  }
  Eigen::VectorXd r, s, w_new;

  f = fun_obj_->get_func(w_);
  grad_.noalias() = fun_obj_->get_grad(w_);

  delta = grad_.norm();
  gnorm_old = delta;
  grad_norm_ = gnorm_old;

  while (outer_iteration < max_iteration) {
    s = tron_cg(delta, grad_, r, cg_iteration, grad_norm_);
    w_new.noalias() = w_ + s;
    f_new = fun_obj_->get_func(w_new);
    actual_reduction = f - f_new;
    gts = grad_.transpose() * s;
    model_reduction = -0.5 * (gts - (s.transpose() * r));
    snorm = s.norm();

    if (outer_iteration == 1)
      delta = std::min(delta, snorm);
    alpha = 0;
    if (f_new - f - gts <= 0)
      alpha = sigma3;
    else
      alpha = std::max(sigma1, -0.5 * (gts / (f_new - f - gts)));

    if (actual_reduction < eta0 * model_reduction)
      delta = std::min(std::max(alpha, sigma1) * snorm, sigma2 * delta);
    else if (actual_reduction < eta1 * model_reduction)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma2 * delta));
    else if (actual_reduction < eta2 * model_reduction)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma3 * delta));
    else
      delta = std::max(delta, std::min(alpha * snorm, sigma3 * delta));

    if (actual_reduction > eta0 * model_reduction) {
      w_.noalias() = w_new;
      grad_ = fun_obj_->get_grad(w_);
      grad_norm_ = grad_.norm();

      count_rejecting_step = 0;
      if (grad_norm_ <= stopping_criterion) {
        break;
      }
      f = f_new;
      gnorm_old = grad_norm_;
      ++outer_iteration;
    } else {
      ++count_rejecting_step;
      // fun_obj_->getFunc(w);
    }
    if (std::isnan(f_new) || (grad_norm_ <= stopping_criterion) ||
        (fabs(actual_reduction) <= 0 && model_reduction <= 0) ||
        (fabs(actual_reduction) <= 1.0e-16 * fabs(f) &&
         fabs(model_reduction) <= 1.0e-16 * fabs(f)) ||
        (std::isinf(delta))) {
      break;
    }
  }
  return w_;
}

Eigen::VectorXd Tron::train_warm_start_inexact(const Eigen::VectorXd &w_start,
                                               const double inexact_level,
                                               int &num_ub, int &num_lb) {
  double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;
  double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4.0;

  int outer_iteration = 1, cg_iteration = 0, count_rejecting_step = 0;
  double delta, snorm, f = 0.0, f_new = 0.0, actual_reduction, model_reduction,
                       gts, gnorm_old, alpha, ub_validerr, lb_validerr;

  int n = fun_obj_->get_variable();
  w_ = w_start;
  if (n != w_start.size())
    w_.conservativeResize(n);

  Eigen::VectorXd r, s, w_new;

  f = fun_obj_->get_func(w_);
  grad_.noalias() = fun_obj_->get_grad(w_);

  delta = grad_.norm();
  gnorm_old = delta;
  grad_norm_ = gnorm_old;

  get_ub_lb_ve_apprx(num_ub, num_lb);
  ub_validerr = static_cast<double>(num_ub) / valid_l_;
  lb_validerr = static_cast<double>(num_lb) / valid_l_;
  if ((ub_validerr - lb_validerr) < inexact_level) {
    // std::cout << "TRON : " << std::scientific << ub_validerr - lb_validerr
    //           << " " << inexact_level << std::fixed << std::endl;
    return w_;
  }
  while (outer_iteration < max_iteration ||
         count_rejecting_step < max_iteration) {
    s = tron_cg(delta, grad_, r, cg_iteration, grad_norm_);
    w_new = w_ + s;
    f_new = fun_obj_->get_func(w_new);
    actual_reduction = f - f_new;
    gts = grad_.transpose() * s;
    model_reduction = -0.5 * (gts - (s.transpose() * r));
    snorm = s.norm();

    if (outer_iteration == 1)
      delta = std::min(delta, snorm);
    alpha = 0;
    if (f_new - f - gts <= 0)
      alpha = sigma3;
    else
      alpha = std::max(sigma1, -0.5 * (gts / (f_new - f - gts)));

    if (actual_reduction < eta0 * model_reduction)
      delta = std::min(std::max(alpha, sigma1) * snorm, sigma2 * delta);
    else if (actual_reduction < eta1 * model_reduction)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma2 * delta));
    else if (actual_reduction < eta2 * model_reduction)
      delta = std::max(sigma1 * delta, std::min(alpha * snorm, sigma3 * delta));
    else
      delta = std::max(delta, std::min(alpha * snorm, sigma3 * delta));

    if (actual_reduction > eta0 * model_reduction) {
      w_.noalias() = w_new;
      grad_ = fun_obj_->get_grad(w_);
      grad_norm_ = grad_.norm();
      count_rejecting_step = 0;

      if (grad_norm_ < 1.0) {
        get_ub_lb_ve_apprx(num_ub, num_lb);
        ub_validerr = static_cast<double>(num_ub) / valid_l_;
        lb_validerr = static_cast<double>(num_lb) / valid_l_;
        // std::cout << "TRON : " << std::scientific << ub_validerr -
        // lb_validerr
        //           << " " << grad_norm_ << " " << ub_validerr << " "
        //           << lb_validerr << " " << grad_norm_ << std::fixed
        //           << std::endl;
        if (ub_validerr - lb_validerr <= inexact_level)
          break;
      }

      f = f_new;
      gnorm_old = grad_norm_;
      ++outer_iteration;
    } else {
      ++count_rejecting_step;
    }
    if (std::isnan(f_new) || (grad_norm_ <= stopping_criterion) ||
        std::isnan(actual_reduction) ||
        (fabs(actual_reduction) <= 0 && model_reduction <= 0) ||
        (fabs(actual_reduction) <= 1.0e-16 * fabs(f) &&
         fabs(model_reduction) <= 1.0e-16 * fabs(f)) ||
        (std::isinf(delta))) {
      break;
    }
  }
  if (valid_n_ != w_.size()) {
    w_.conservativeResize(valid_n_);
    grad_.conservativeResize(valid_n_);
  }
  return w_;
}

Eigen::VectorXd Tron::tron_cg(const double delta, const Eigen::VectorXd &g,
                              Eigen::VectorXd &r, int &cg_iteration,
                              double gnorm) {
  int n = fun_obj_->get_variable();
  double xig, rtr, tmprtr, alpha, std;
  Eigen::VectorXd d, Hd;
  Eigen::VectorXd s = Eigen::VectorXd::Zero(n);

  r.noalias() = -g;
  d.noalias() = r;

  if (gnorm < 10)
    xig = 0.01 * gnorm;
  else
    xig = 0.1 * gnorm;
  // xig = 0.1 * gnorm;

  rtr = r.transpose() * r;
  cg_iteration = 0;
  while (1) {
    ++cg_iteration;
    if (r.norm() <= xig) {
      break;
    }

    Hd = fun_obj_->product_hesse_vec(d);

    alpha = rtr / (d.transpose() * Hd);
    s.noalias() += alpha * d;
    if (s.norm() > delta) {
      // std::cout << "cg reaches the trust region boundary" <<std::endl;
      s.noalias() -= alpha * d;
      std = s.transpose() * d;
      if (std >= 0) {
        alpha = (delta * delta - s.transpose() * s) /
                (std +
                 sqrt(std * std + ((d.transpose() * d) *
                                   (delta * delta - (s.transpose() * s)))(0)));
      } else {
        alpha = (sqrt(std * std + ((d.transpose() * d) *
                                   (delta * delta - (s.transpose() * s)))(0)) -
                 std) /
                (d.transpose() * d);
      }

      s.noalias() += alpha * d;
      r.noalias() -= alpha * Hd;
      break;
    }
    r.noalias() -= alpha * Hd;
    tmprtr = rtr;
    rtr = r.transpose() * r;
    d = r + ((rtr / tmprtr) * d).matrix();
  }
  return s;
}

double Tron::get_valid_error(const Eigen::VectorXd &w) {
  Eigen::VectorXd vw = w;
  if (valid_n_ != w.size()) {
    vw.conservativeResize(valid_n_);
  }
  Eigen::ArrayXd prob = valid_y_ * (valid_x_ * vw).array();
  double miss = 0;
  for (int i = 0; i < valid_l_; ++i) {
    if (prob.coeffRef(i) < 0.0)
      ++miss;
  }

  return miss / valid_l_;
}

void Tron::get_ub_lb_ve_apprx(int &ub_ve, int &lb_ve) {
  Eigen::VectorXd vw = w_;
  Eigen::VectorXd vgrad = grad_;
  if (vw.size() != valid_n_) {
    vw.conservativeResize(valid_n_);
    vgrad.conservativeResize(valid_n_);
  }
  Eigen::VectorXd w_x = valid_x_ * vw;
  Eigen::VectorXd g_x = valid_x_ * vgrad;
  double g_norm = grad_.norm();

  ub_ve = 0;
  lb_ve = 0;
  double w_x_i, g_x_i, gxi_norm, ub_wstar_xi, lb_wstar_xi;
  for (int i = 0; i < valid_l_; ++i) {
    w_x_i = w_x.coeffRef(i);
    g_x_i = g_x.coeffRef(i);
    gxi_norm = g_norm * valid_x_norm_.coeffRef(i);
    ub_wstar_xi = w_x_i - 0.5 * (g_x_i - gxi_norm);
    lb_wstar_xi = w_x_i - 0.5 * (g_x_i + gxi_norm);
    // std::cout<< ub_wstar_xi <<" " <<lb_wstar_xi << std::endl;
    if (valid_y_.coeffRef(i) > 0.0) {
      if (ub_wstar_xi < 0.0)
        ++lb_ve;
      if (lb_wstar_xi < 0.0)
        ++ub_ve;
    } else {
      if (lb_wstar_xi > 0.0)
        ++lb_ve;
      if (ub_wstar_xi > 0.0)
        ++ub_ve;
    }
  }
  // std::cout <<"  " << ub_ve - lb_ve <<" " <<ub_ve <<" " <<lb_ve <<std::endl;
}

std::vector<double> Tron::get_c_set_right_opt(const double &now_c,
                                              const Eigen::VectorXd &w_star,
                                              int &miss) {
  miss = 0;
  Eigen::VectorXd vw = w_star;
  if (w_star.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();
  double vw_norm = vw.norm();
  double xi_norm, zi;
  std::vector<double> c_vec;
  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (valid_y_.coeffRef(i) * zi < 0.0) {
      ++miss;
      if (zi < 0.0) {
        c_vec.push_back(now_c * (vw_norm * xi_norm - zi) /
                        (vw_norm * xi_norm + zi));
      } else if (zi > 0.0) {
        c_vec.push_back(now_c * (vw_norm * xi_norm + zi) /
                        (vw_norm * xi_norm - zi));
      }
    }
  }
  std::sort(c_vec.begin(), c_vec.end());
  return c_vec;
}

std::vector<double> Tron::get_c_set_left_subopt(const bool &flag_sort) const  {
  double now_c = fun_obj_->get_regularized_parameter();
  Eigen::VectorXd w_x = valid_x_ * w_;
  Eigen::VectorXd g_x = valid_x_ * grad_;

  std::vector<double> c_vec;
  double w_norm = w_.norm();
  double g_norm = grad_.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0 && (wxi + 0.5 * (gxi_norm - gxi)) < 0.0) {
      c_vec.push_back(now_c * (wxi_norm + wxi) /
                      (wxi_norm - wxi + gxi_norm + gxi));
    } else if (valid_y_.coeffRef(i) < 0.0 &&
               (wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
      c_vec.push_back(now_c * (wxi_norm - wxi) /
                      (wxi_norm + wxi + gxi_norm - gxi));
    }
  }
  if (flag_sort)
    std::sort(c_vec.begin(), c_vec.end(), std::greater<double>());
  return c_vec;
}

std::vector<double> Tron::get_c_set_right_subopt(const bool &flag_sort) const {
  double now_c = fun_obj_->get_regularized_parameter();
  Eigen::VectorXd w_x = valid_x_ * w_;
  Eigen::VectorXd g_x = valid_x_ * grad_;

  std::vector<double> c_vec;
  double w_norm = w_.norm();
  double g_norm = grad_.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0 && (wxi + 0.5 * (gxi_norm - gxi)) < 0.0) {
      c_vec.push_back(now_c * (wxi_norm - wxi) /
                      (wxi_norm + wxi + gxi_norm - gxi));
    } else if (valid_y_.coeffRef(i) < 0.0 &&
               (wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
      c_vec.push_back(now_c * (wxi_norm + wxi) /
                      (wxi_norm - wxi + gxi_norm + gxi));
    }
  }
  if (flag_sort)
    std::sort(c_vec.begin(), c_vec.end());
  return c_vec;
}

std::vector<double> Tron::get_c_set_left_subopt(const double &now_c,
                                                const Eigen::VectorXd &w,
                                                const Eigen::VectorXd &grad_w,
                                                const bool &flag_sort) const {
  Eigen::VectorXd w_x = valid_x_ * w;
  Eigen::VectorXd g_x = valid_x_ * grad_w;

  std::vector<double> c_vec;
  double w_norm = w.norm();
  double g_norm = grad_w.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0 && (wxi + 0.5 * (gxi_norm - gxi)) < 0.0) {
      c_vec.push_back(now_c * (wxi_norm + wxi) /
                      (wxi_norm - wxi + gxi_norm + gxi));
    } else if (valid_y_.coeffRef(i) < 0.0 &&
               (wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
      c_vec.push_back(now_c * (wxi_norm - wxi) /
                      (wxi_norm + wxi + gxi_norm - gxi));
    }
  }
  if (flag_sort)
    std::sort(c_vec.begin(), c_vec.end(), std::greater<double>());
  return c_vec;
}

std::vector<double> Tron::get_c_set_right_subopt(const double &now_c,
                                                 const Eigen::VectorXd &w,
                                                 const Eigen::VectorXd &grad_w,
                                                 const bool &flag_sort) const {
  Eigen::VectorXd w_x = valid_x_ * w;
  Eigen::VectorXd g_x = valid_x_ * grad_w;

  std::vector<double> c_vec;
  double w_norm = w.norm();
  double g_norm = grad_w.norm();
  double wxi, gxi, wxi_norm, gxi_norm;

  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;

    if (valid_y_.coeffRef(i) > 0.0 && (wxi + 0.5 * (gxi_norm - gxi)) < 0.0) {
      c_vec.push_back(now_c * (wxi_norm - wxi) /
                      (wxi_norm + wxi + gxi_norm - gxi));
    } else if (valid_y_.coeffRef(i) < 0.0 &&
               (wxi - 0.5 * (gxi + gxi_norm)) > 0.0) {
      c_vec.push_back(now_c * (wxi_norm + wxi) /
                      (wxi_norm - wxi + gxi_norm + gxi));
    }
  }
  if (flag_sort)
    std::sort(c_vec.begin(), c_vec.end());
  return c_vec;
}

std::vector<double>
Tron::get_c_set_right_opt_for_path(const Eigen::VectorXd &w_star,
                                   const double &now_c) {
  Eigen::VectorXd vw = w_star;
  if (w_star.size() != valid_n_)
    vw.conservativeResize(valid_n_);
  Eigen::ArrayXd z = (valid_x_ * vw).array();
  double vw_norm = vw.norm();
  double xi_norm, zi;
  std::vector<double> c_vec;
  for (int i = 0; i < valid_l_; ++i) {
    zi = z.coeffRef(i);
    xi_norm = valid_x_norm_.coeffRef(i);
    if (zi < 0.0) {
      c_vec.push_back(now_c * (vw_norm * xi_norm - zi) /
                      (vw_norm * xi_norm + zi));
    } else if (zi > 0.0) {
      c_vec.push_back(now_c * (vw_norm * xi_norm + zi) /
                      (vw_norm * xi_norm - zi));
    }
  }
  return c_vec;
}

std::vector<double>
Tron::get_c_set_right_subopt_for_path(const Eigen::VectorXd &w,
                                      const Eigen::VectorXd &grad_w,
                                      const double &now_c, int &num_dif_ublb) {

  Eigen::VectorXd w_x = valid_x_ * w;
  Eigen::VectorXd g_x = valid_x_ * grad_w;

  std::vector<double> c_vec;
  double w_norm = w.norm();
  double g_norm = grad_w.norm();
  double wxi, gxi, wxi_norm, gxi_norm, ub_wxi, lb_wxi;
  int num_lb = 0, num_ub = 0;
  // int tmp_ub = 0;
  for (int i = 0; i < valid_l_; ++i) {
    wxi = w_x.coeffRef(i);
    gxi = g_x.coeffRef(i);
    wxi_norm = valid_x_norm_.coeffRef(i) * w_norm;
    gxi_norm = valid_x_norm_.coeffRef(i) * g_norm;
    ub_wxi = wxi + 0.5 * (gxi_norm - gxi);
    lb_wxi = wxi - 0.5 * (gxi_norm + gxi);

    double ub_to_zero =
        now_c * (wxi_norm - wxi) / (wxi_norm + wxi + gxi_norm - gxi);
    double lb_to_zero =
        now_c * (wxi_norm + wxi) / (wxi_norm - wxi + gxi_norm + gxi);
    double yi = valid_y_.coeffRef(i);

    // about cal{P} and cal{N}
    if (yi > 0.0 && ub_wxi < 0.0) {
      ++num_lb;
      c_vec.push_back(ub_to_zero);
    }
    if (yi < 0.0 && lb_wxi > 0.0) {
      ++num_lb;
      c_vec.push_back(lb_to_zero);
    }

    // about cal{P'} and cal{N'}
    if (valid_x_norm_.coeffRef(i) != 0.0) {
      if (yi > 0.0 && lb_wxi >= 0.0) {
        ++num_ub;
        c_vec.push_back(lb_to_zero);
      }
      if (yi < 0.0 && ub_wxi <= 0.0) {
        ++num_ub;
        c_vec.push_back(ub_to_zero);
      }
    } else {
      ++num_ub;
    }
  }
  // std::cout << "TRON check " << valid_l_ - tmp_ub << " " << num_ub << " "
  //           << num_ub - num_lb << std::endl;
  num_ub = valid_l_ - num_ub;
  // std::cout << "ub " << num_ub <<" , lb " << num_lb <<" "
  // <<(double)(num_ub-num_lb)/valid_l_  <<std::endl;
  num_dif_ublb = num_ub - num_lb;
  return c_vec;
}

} // namespace sdm
