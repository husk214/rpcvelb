#include "L2R_Huber_SVC.h"

L2R_Huber_SVC::L2R_Huber_SVC(const std::string fileNameLibSVMFormat,
                             const double c)
    : X(), y(), l(), n(), C(c), z(), indexs_D1(), indexs_D2(), XD2w_yD2() {
  read_LibSVMdata1(fileNameLibSVMFormat.c_str(), X, y);
  l = X.rows();
  n = X.cols();
  z = Eigen::ArrayXd::Zero(l);
  XD2w_yD2 = Eigen::VectorXd::Zero(l);
}

L2R_Huber_SVC::L2R_Huber_SVC(
    const Eigen::SparseMatrix<double, 1, std::ptrdiff_t> train_x,
    const Eigen::ArrayXd train_y, const double c)
    : X(train_x), y(train_y), l(), n(), C(c), z(), indexs_D1(), indexs_D2(), XD2w_yD2() {
  l = X.rows();
  n = X.cols();
  z = Eigen::ArrayXd::Zero(l);
  XD2w_yD2 = Eigen::VectorXd::Zero(l);
}

L2R_Huber_SVC::~L2R_Huber_SVC() {}

L2R_Huber_SVC *
L2R_Huber_SVC::create(const std::string fileNameLibSVMFormat) const {
  return new L2R_Huber_SVC(fileNameLibSVMFormat);
}

Eigen::VectorXd L2R_Huber_SVC::productXVsub2(const Eigen::VectorXd &V) {
  Eigen::VectorXd ans;
  int sizeD = indexs_D2.size();
  ans = Eigen::VectorXd::Zero(sizeD);

  for (int i = 0; i < sizeD; ++i) {
    for (Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator it(
             X, indexs_D2[i]);
         it; ++it) {
      ans.coeffRef(i) += V.coeffRef(it.index()) * it.value();
    }
  }
  return ans;
}

Eigen::VectorXd L2R_Huber_SVC::productXtVsub1(const Eigen::VectorXd &V) {
  Eigen::VectorXd ans = Eigen::VectorXd::Zero(n);
  int sizeD = indexs_D1.size();
  int indexD1;
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

Eigen::VectorXd L2R_Huber_SVC::productXtVsub2(const Eigen::VectorXd &V) {
  Eigen::VectorXd ans = Eigen::VectorXd::Zero(n);
  int sizeD = indexs_D2.size();
  for (int i = 0; i < sizeD; ++i) {
    for (Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator it(
             X, indexs_D2[i]);
         it; ++it) {
      ans.coeffRef(it.index()) += V.coeffRef(i) * it.value();
    }
  }
  return ans;
}

double L2R_Huber_SVC::get_func(const Eigen::VectorXd &w) {
  z = y * (X * w).array();
  int indexD2 = 0;
  double zi, tmp;
  double f = 0;
  indexs_D1.clear();
  indexs_D2.clear();
  XD2w_yD2 = Eigen::VectorXd::Zero(l);

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

Eigen::VectorXd L2R_Huber_SVC::get_grad(const Eigen::VectorXd &w) {
  Eigen::VectorXd XD1tyD1 = productXtVsub1(y.matrix());
  Eigen::VectorXd XD2tXD2w_yD2 = productXtVsub2(XD2w_yD2);
  return w + C * (XD2tXD2w_yD2 - XD1tyD1);
}

Eigen::VectorXd L2R_Huber_SVC::get_loss_grad(const Eigen::VectorXd &w) {
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

Eigen::VectorXd L2R_Huber_SVC::product_hesse_vec(const Eigen::VectorXd &V) {
  Eigen::VectorXd XD2V = productXVsub2(V);
  Eigen::VectorXd XD2tXD2V = productXtVsub2(XD2V);
  return V + C * XD2tXD2V;
}

int L2R_Huber_SVC::get_variable(void) { return n; }
double L2R_Huber_SVC::get_regularized_parameter(void) { return C; }
void L2R_Huber_SVC::set_regularized_parameter(const double &c) { C = c; }

double L2R_Huber_SVC::predict(const std::string fileNameLibSVMFormat,
                              Eigen::VectorXd &w) {
  Eigen::SparseMatrix<double, 1> vX;
  Eigen::ArrayXd vy;
  const char *fn = fileNameLibSVMFormat.c_str();
  read_LibSVMdata1(fn, vX, vy);
  // loadLib(fileNameLibSVMFormat, vX, vy);
  int vl = vX.rows();
  int vn = vX.cols();
  int success = 0;

  if (vn != w.size()) {
    w.conservativeResize(vn);
  }
  Eigen::ArrayXd prob = vy * (vX * w).array();

  for (int i = 0; i < vl; ++i) {
    if (prob.coeffRef(i) >= 0.0)
      ++success;
  }
  return (double)success / vl;
}
