#include "Ctgp.h"

namespace sdm {

Ctgp::Ctgp(Primal_solver *sol_obj, const double &inexact_level,
           const double c_min, const double c_max, const double min_move_c)
    : psol_obj_(sol_obj), dsol_obj_(nullptr), inexact_level_(inexact_level),
      c_min_(c_min), c_max_(c_max), min_move_c_(min_move_c), train_l_(),
      valid_l_(), best_miss_(), best_valid_err_(), worst_lb_(1.0),
      worst_lb_c_(c_min), epsilon_(1.0), psol_objs_(), fold_num_(0),
      whole_l_(0) {
  valid_l_ = psol_obj_->get_valid_l();
  best_miss_ = valid_l_;
}

Ctgp::Ctgp(std::vector<Primal_solver *> psol_objs, const double &inexact_level,
           const double c_min, const double c_max, const double min_move_c)
    : psol_obj_(), dsol_obj_(nullptr), inexact_level_(inexact_level),
      c_min_(c_min), c_max_(c_max), min_move_c_(min_move_c), train_l_(),
      valid_l_(), best_miss_(), best_valid_err_(), worst_lb_(1.0),
      worst_lb_c_(c_min), epsilon_(1.0), psol_objs_(psol_objs), fold_num_(),
      whole_l_() {
  fold_num_ = psol_objs_.size();
  for (int i = 0; i < fold_num_; ++i)
    whole_l_ += (psol_objs_[i])->get_valid_l();
  best_miss_ = whole_l_;
  valid_l_ = whole_l_;
}

Ctgp::Ctgp(Dual_solver *sol_obj, const double &inexact_level,
           const double c_min, const double c_max, const double min_move_c)
    : psol_obj_(nullptr), dsol_obj_(sol_obj), inexact_level_(inexact_level),
      c_min_(c_min), c_max_(c_max), min_move_c_(min_move_c), train_l_(),
      valid_l_(), best_miss_(), best_valid_err_(), worst_lb_(1.0),
      worst_lb_c_(c_min), epsilon_(1.0), psol_objs_(), fold_num_(0),
      whole_l_(0) {
  train_l_ = dsol_obj_->get_train_l();
  valid_l_ = dsol_obj_->get_valid_l();
  best_miss_ = valid_l_;
}

Ctgp::~Ctgp() {}

void Ctgp::update_best(const int &num_ub) {
  if (best_miss_ > num_ub) {
    best_miss_ = num_ub;
    best_valid_err_ = (double)best_miss_ / (double)valid_l_;
    epsilon_ = best_valid_err_ - worst_lb_;
  }
}

void Ctgp::update_best(const double &valid_err) {
  if (best_valid_err_ > valid_err) {
    best_valid_err_ = valid_err;
    epsilon_ = best_valid_err_ - worst_lb_;
  }
}

void Ctgp::update_worst_lbve(const double &lbve) {
  if (worst_lb_ > lbve) {
    worst_lb_ = lbve;
    epsilon_ = best_valid_err_ - worst_lb_;
  }
}

void Ctgp::update_worst_lbve(const double &lbve, const double &c_key) {
  if (worst_lb_ > lbve) {
    worst_lb_ = lbve;
    worst_lb_c_ = c_key;
    epsilon_ = best_valid_err_ - worst_lb_;
  }
}

void Ctgp::update_epsilon(const double &control_worst_lb_first) {
  worst_lb_ = control_worst_lb_first;
  epsilon_ = best_valid_err_ - worst_lb_;
}


double Ctgp::find_worst_lb(const double &c1, const double &c2,
                           const Eigen::VectorXd &w_c2,
                           const Eigen::VectorXd &grad_c2) const {
  std::vector<double> c_vec_lb;
  double sub_worst_lb = 0.0;
  if (c1 < c2) {
    c_vec_lb = psol_obj_->get_c_set_left_subopt(c2, w_c2, grad_c2);
    for (std::size_t i = 0; i != c_vec_lb.size(); ++i) {
      if (c_vec_lb[i] <= c1) {
        sub_worst_lb = static_cast<double>(static_cast<int>(c_vec_lb.size()) -
                                           static_cast<int>(i)) /
                       valid_l_;
        break;
      }
    }
  } else {
    c_vec_lb = psol_obj_->get_c_set_right_subopt(c2, w_c2, grad_c2);
    for (std::size_t i = 0; i != c_vec_lb.size(); ++i) {
      if (c_vec_lb[i] >= c1) {
        sub_worst_lb = static_cast<double>(static_cast<int>(c_vec_lb.size()) -
                                           static_cast<int>(i)) /
                       valid_l_;
        break;
      }
    }
  }
  return sub_worst_lb;
}

double Ctgp::find_worst_lb(const double &c1, const Eigen::VectorXd &w_c1,
                           const Eigen::VectorXd &grad_c1, const double &c2,
                           const Eigen::VectorXd &w_c2,
                           const Eigen::VectorXd &grad_c2) const {
  // assume c1 < c2
  std::vector<double> c1_lb_right, c2_lb_left;
  double sub_worst_lb = 0.0;
  c1_lb_right = psol_obj_->get_c_set_right_subopt(c1, w_c1, grad_c1);
  c2_lb_left = psol_obj_->get_c_set_left_subopt(c2, w_c2, grad_c2);
  int miss1 = c1_lb_right.size(), miss2 = c2_lb_left.size();
  std::size_t miss_diff = static_cast<std::size_t>(abs(miss1 - miss2));
  bool flag_intersect = true;
  if (miss1 < miss2) {
    for (std::size_t pre_i = 0; pre_i < miss_diff; ++pre_i) {
      if (c1 >= c2_lb_left[pre_i]) {
        flag_intersect = false;
        sub_worst_lb =
            static_cast<double>(miss2 - static_cast<int>(pre_i)) / valid_l_;
        break;
      }
    }
    if (flag_intersect) {
      for (std::size_t i = 0; i != c1_lb_right.size(); ++i, ++miss_diff) {
        if (c1_lb_right[i] >= c2_lb_left[miss_diff]) {
          sub_worst_lb =
              static_cast<double>(miss1 - static_cast<int>(i)) / valid_l_;
          break;
        }
      }
    }
  } else {
    for (std::size_t pre_i = 0; pre_i < miss_diff; ++pre_i) {
      if (c2 <= c1_lb_right[pre_i]) {
        flag_intersect = false;
        sub_worst_lb =
            static_cast<double>(miss1 - static_cast<int>(pre_i)) / valid_l_;
        break;
      }
    }
    if (flag_intersect) {
      for (std::size_t i = 0; i != c2_lb_left.size(); ++i, ++miss_diff) {
        if (c2_lb_left[i] <= c1_lb_right[miss_diff]) {
          sub_worst_lb =
              static_cast<double>(miss2 - static_cast<int>(i)) / valid_l_;
          break;
        }
      }
    }
  }
  return sub_worst_lb;
}

double Ctgp::find_worst_lb(const double &c1, const double &c2,
                           const std::vector<double> &c2_lb) const {
  double sub_worst_lb = 0.0;
  int miss = static_cast<int>(c2_lb.size());
  if (c1 < c2) {
    for (std::size_t i = 0; i != c2_lb.size(); ++i) {
      if (c2_lb[i] <= c1) {
        sub_worst_lb =
            static_cast<double>(miss - static_cast<int>(i)) / valid_l_;
        break;
      }
    }
  } else {
    for (std::size_t i = 0; i != c2_lb.size(); ++i) {
      if (c2_lb[i] >= c1) {
        sub_worst_lb =
            static_cast<double>(miss - static_cast<int>(i)) / valid_l_;
        break;
      }
    }
  }
  return sub_worst_lb;
}
double Ctgp::find_worst_lb(const double &c1,
                           const std::vector<double> &c1_lb_right,
                           const double &c2,
                           const std::vector<double> &c2_lb_left) const {
  // assume c1 < c2
  double sub_worst_lb = 0.0;
  int miss1 = static_cast<int>(c1_lb_right.size());
  int miss2 = static_cast<int>(c2_lb_left.size());

  std::size_t miss_diff = static_cast<std::size_t>(abs(miss1 - miss2));
  bool flag_intersect = true;
  if (miss1 < miss2) {
    for (std::size_t pre_i = 0; pre_i < miss_diff; ++pre_i) {
      if (c1 >= c2_lb_left[pre_i]) {
        flag_intersect = false;
        sub_worst_lb =
            static_cast<double>(miss2 - static_cast<int>(pre_i)) / valid_l_;
        break;
      }
    }
    if (flag_intersect) {
      for (std::size_t i = 0; i != c1_lb_right.size(); ++i, ++miss_diff) {
        if (c1_lb_right[i] >= c2_lb_left[miss_diff]) {
          sub_worst_lb =
              static_cast<double>(miss1 - static_cast<int>(i)) / valid_l_;
          break;
        }
      }
    }
  } else {
    for (std::size_t pre_i = 0; pre_i < miss_diff; ++pre_i) {
      if (c2 <= c1_lb_right[pre_i]) {
        flag_intersect = false;
        sub_worst_lb =
            static_cast<double>(miss1 - static_cast<int>(pre_i)) / valid_l_;
        break;
      }
    }
    if (flag_intersect) {
      for (std::size_t i = 0; i != c2_lb_left.size(); ++i, ++miss_diff) {
        if (c2_lb_left[i] <= c1_lb_right[miss_diff]) {
          sub_worst_lb =
              static_cast<double>(miss2 - static_cast<int>(i)) / valid_l_;
          break;
        }
      }
    }
  }
  return sub_worst_lb;
}

double Ctgp::find_worst_lb(const double &c1,
                           const std::vector<double> &c1_lb_right,
                           const double &c2,
                           const std::vector<double> &c2_lb_left,
                           double &midpoint) const {
  // assume c1 < c2
  double sub_worst_lb = 0.0;
  int miss1 = static_cast<int>(c1_lb_right.size());
  int miss2 = static_cast<int>(c2_lb_left.size());
  midpoint = 0.5 * (c1 + c2);
  std::size_t miss_diff = static_cast<std::size_t>(abs(miss1 - miss2));
  bool flag_intersect = true;
  if (miss1 < miss2) {
    for (std::size_t pre_i = 0; pre_i < miss_diff; ++pre_i) {
      if (c1 >= c2_lb_left[pre_i]) {
        flag_intersect = false;
        if (pre_i > 0) {
          midpoint = pow(10, (log10(c1)+log10(c2_lb_left[pre_i-1]))*0.5);
        } else {
          midpoint = pow(10, (log10(c1)+log10(c2))*0.5);
        }
        sub_worst_lb =
            static_cast<double>(miss2 - static_cast<int>(pre_i)) / valid_l_;
        break;
      }
    }
    if (flag_intersect) {
      for (std::size_t i = 0; i != c1_lb_right.size(); ++i, ++miss_diff) {
        if (c1_lb_right[i] >= c2_lb_left[miss_diff]) {
          sub_worst_lb =
              static_cast<double>(miss1 - static_cast<int>(i)) / valid_l_;
          break;
        }
      }
    }
  } else {
    for (std::size_t pre_i = 0; pre_i < miss_diff; ++pre_i) {
      if (c2 <= c1_lb_right[pre_i]) {
        flag_intersect = false;
        if (pre_i > 0) {
          midpoint = pow(10, (log10(c2)+log10(c1_lb_right[pre_i-1]))*0.5);
        } else {
          midpoint = pow(10, (log10(c1)+log10(c2))*0.5);
        }
        sub_worst_lb =
            static_cast<double>(miss1 - static_cast<int>(pre_i)) / valid_l_;
        break;
      }
    }
    if (flag_intersect) {
      for (std::size_t i = 0; i != c2_lb_left.size(); ++i, ++miss_diff) {
        if (c2_lb_left[i] <= c1_lb_right[miss_diff]) {
          sub_worst_lb =
              static_cast<double>(miss2 - static_cast<int>(i)) / valid_l_;
          break;
        }
      }
    }
  }
  return sub_worst_lb;
}

} // naespace sdm
