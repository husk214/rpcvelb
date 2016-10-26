#include "L2r_huber_svc.h"

namespace sdm {
L2r_huber_svc::L2r_huber_svc(const std::string fileNameLibSVMFormat,
                             const double c)
    : X(), y(), l(), n(), C(c), z(), indexs_D1(), indexs_D2(), XD2w_yD2() {
  sdm::load_libsvm_binary(X, y, fileNameLibSVMFormat);
  l = X.rows();
  n = X.cols();
  z = Eigen::ArrayXd::Zero(l);
  XD2w_yD2 = Eigen::VectorXd::Zero(l);
}

L2r_huber_svc::L2r_huber_svc(
    const Eigen::SparseMatrix<double, 1, std::ptrdiff_t> train_x,
    const Eigen::ArrayXd train_y, const double c)
    : X(train_x), y(train_y), l(), n(), C(c), z(), indexs_D1(), indexs_D2(), XD2w_yD2() {
  l = X.rows();
  n = X.cols();
  z = Eigen::ArrayXd::Zero(l);
  XD2w_yD2 = Eigen::VectorXd::Zero(l);
}

L2r_huber_svc::~L2r_huber_svc() {}

Eigen::VectorXd L2r_huber_svc::productXVsub2(const Eigen::VectorXd &V) {
  Eigen::VectorXd ans;
  int sizeD = indexs_D2.size();
  ans = Eigen::VectorXd::Zero(sizeD);

// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
  for (int i = 0; i < sizeD; ++i) {
    for (Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator it(
             X, indexs_D2[i]);
         it; ++it) {
      ans.coeffRef(i) += V.coeffRef(it.index()) * it.value();
    }
  }
  return ans;
}

Eigen::VectorXd L2r_huber_svc::productXtVsub1(const Eigen::VectorXd &V) {
  Eigen::VectorXd ans = Eigen::VectorXd::Zero(n);
  int sizeD = indexs_D1.size();
  int indexD1;

#ifdef _OPENMP
#pragma omp parallel for
#endif
  for (int i = 0; i < sizeD; ++i) {
    indexD1 = indexs_D1[i];
    for (Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator it(
             X, indexD1);
         it; ++it) {
      ans.coeffRef(it.index()) += V.coeffRef(indexD1) * it.value();
    }
  }
  return ans;
}

Eigen::VectorXd L2r_huber_svc::productXtVsub2(const Eigen::VectorXd &V) {
  Eigen::VectorXd ans = Eigen::VectorXd::Zero(n);
  int sizeD = indexs_D2.size();

// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
  for (int i = 0; i < sizeD; ++i) {
    for (Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator it(
             X, indexs_D2[i]);
         it; ++it) {
      ans.coeffRef(it.index()) += V.coeffRef(i) * it.value();
    }
  }
  return ans;
}

double L2r_huber_svc::get_func(const Eigen::VectorXd &w) {
  z = y * (X * w).array();
  int indexD2 = 0;
  double zi, tmp;
  double f = 0.0;
  indexs_D1.clear();
  indexs_D2.clear();
  XD2w_yD2 = Eigen::VectorXd::Zero(l);

// #ifdef _OPENMP
// #pragma omp parallel for
// #endif
  for (int i = 0; i < l; ++i) {
    zi = z.coeffRef(i);
    tmp = 1 - zi;
    if (zi <= 0) {
      indexs_D1.push_back(i);
      f += (tmp - 0.5);
    } else if (zi < 1) {
      indexs_D2.push_back(i);
      f += tmp * tmp * 0.5;
      XD2w_yD2.coeffRef(indexD2) = -y.coeffRef(i) * tmp;
      ++indexD2;
    }
  }
  f *= C;
  f += w.squaredNorm() / 2.0;
  return f;
}

Eigen::VectorXd L2r_huber_svc::get_grad(const Eigen::VectorXd &w) {
  Eigen::VectorXd XD1tyD1 = productXtVsub1(y.matrix());
  Eigen::VectorXd XD2tXD2w_yD2 = productXtVsub2(XD2w_yD2);
  return w + C * (XD2tXD2w_yD2 - XD1tyD1);
}

Eigen::VectorXd L2r_huber_svc::get_loss_grad(const Eigen::VectorXd &w) {
  z = y * (X * w).array();
  int indexD2 = 0;
  double zi, tmp;
  indexs_D1.clear();
  indexs_D2.clear();
  XD2w_yD2 = Eigen::VectorXd::Zero(l);

  for (int i = 0; i < l; ++i) {
    zi = z.coeffRef(i);
    tmp = 1 - zi;
    if (zi <= 0) {
      indexs_D1.push_back(i);
    } else if (zi < 1) {
      indexs_D2.push_back(i);
      XD2w_yD2.coeffRef(indexD2) = -y.coeffRef(i) * tmp;
      ++indexD2;
    }
  }
  Eigen::VectorXd XD1tyD1 = productXtVsub1(y.matrix());
  Eigen::VectorXd XD2tXD2w_yD2 = productXtVsub2(XD2w_yD2);
  return (XD2tXD2w_yD2 - XD1tyD1);
}

Eigen::VectorXd L2r_huber_svc::product_hesse_vec(const Eigen::VectorXd &V) {
  Eigen::VectorXd XD2V = productXVsub2(V);
  Eigen::VectorXd XD2tXD2V = productXtVsub2(XD2V);
  return V + C * XD2tXD2V;
}

int L2r_huber_svc::get_variable(void) { return n; }
double L2r_huber_svc::get_regularized_parameter(void) { return C; }
void L2r_huber_svc::set_regularized_parameter(const double &c) { C = c; }


} //namespace sdm
