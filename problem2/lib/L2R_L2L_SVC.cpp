#include "L2R_L2L_SVC.h"

L2R_L2L_SVC::L2R_L2L_SVC(const std::string fileNameLibSVMFormat, const double c)
    : X(), y(), z(), vI(), l(), n(), C(c) {
  read_LibSVMdata1(fileNameLibSVMFormat.c_str(), X, y);
  l = X.rows();
  n = X.cols();
  z = Eigen::VectorXd::Zero(l);
}

L2R_L2L_SVC::~L2R_L2L_SVC() {}

L2R_L2L_SVC *L2R_L2L_SVC::create(const std::string fileNameLibSVMFormat) const {
  return new L2R_L2L_SVC(fileNameLibSVMFormat);
}

//-- you need using getGrad in advance, because it makes vI --
Eigen::VectorXd L2R_L2L_SVC::productXVsub(const Eigen::VectorXd &V) {
  int sizeI = vI.size();
  Eigen::VectorXd ans = Eigen::VectorXd::Zero(sizeI);
  for (int i = 0; i < sizeI; ++i) {
    for (Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator it(
             X, vI[i]);
         it; ++it) {
      ans.coeffRef(i) += V.coeffRef(it.index()) * it.value();
    }
  }
  return ans;
}

//-- you need using grad in advance, because it makes vI --
Eigen::VectorXd L2R_L2L_SVC::productXtVsub(const Eigen::VectorXd &V) {
  Eigen::VectorXd ans = Eigen::VectorXd::Zero(n);
  int sizeI = vI.size();
  for (int i = 0; i < sizeI; ++i) {
    for (Eigen::SparseMatrix<double, 1, std::ptrdiff_t>::InnerIterator it(
             X, vI[i]);
         it; ++it) {
      ans.coeffRef(it.index()) += V.coeffRef(i) * it.value();
    }
  }
  // std::cout << "XtVsub " << ans <<std::endl;
  return ans;
}

double L2R_L2L_SVC::get_func(const Eigen::VectorXd &w) {
  double f = 0.0;
  z = y * (X * w).array();
  vI.clear();
  for (int i = 0; i < l; ++i) {
    double tmp = 1 - z.coeffRef(i);
    if (tmp > 0) {
      f += C * tmp * tmp;
      vI.push_back(i);
    }
  }
  f *= 2.0;
  f += w.transpose() * w;
  f /= 2.0;
  return f;
}

Eigen::VectorXd L2R_L2L_SVC::get_grad(const Eigen::VectorXd &w) {
  int size_vI = vI.size();
  int v_index;
  for (int i = 0; i < size_vI; ++i) {
    v_index = vI[i];
    z.coeffRef(i) = C * y.coeffRef(v_index) * (z.coeffRef(v_index) - 1);
  }
  return (2.0 * productXtVsub(z) + w);
}

Eigen::VectorXd L2R_L2L_SVC::get_loss_grad(const Eigen::VectorXd &w) {
  z = y * (X * w).array();
  vI.clear();
  for (int i = 0; i < l; ++i) {
    double tmp = 1 - z.coeffRef(i);
    if (tmp > 0) {
      vI.push_back(i);
    }
  }
  int size_vI = vI.size();
  int v_index;
  for (int i = 0; i < size_vI; ++i) {
    v_index = vI[i];
    z.coeffRef(i) = y.coeffRef(v_index) * (z.coeffRef(v_index) - 1);
  }
  return (2.0 * productXtVsub(z));
}

// compute vector producting Hessian(f(w)) and a vector V
// Hessian(f(w))V = V + 2CX_{I,:}^t(X_{I,:}V)
Eigen::VectorXd L2R_L2L_SVC::product_hesse_vec(const Eigen::VectorXd &V) {
  Eigen::VectorXd wa = productXVsub(V);
  Eigen::VectorXd Hv = productXtVsub(wa);
  return V + (2.0 * C) * Hv;
}

int L2R_L2L_SVC::get_variable() { return n; }
double L2R_L2L_SVC::get_regularized_parameter(void) { return C; }
void L2R_L2L_SVC::set_regularized_parameter(const double &c) { C = c; }

double L2R_L2L_SVC::predict(const std::string fileNameLibSVMFormat,
                            Eigen::VectorXd &w) {
  Eigen::SparseMatrix<double, 1> vX;
  Eigen::ArrayXd vy;
  const char *fn = fileNameLibSVMFormat.c_str();
  read_LibSVMdata1(fn, vX, vy);
  // const char *fn = fileNameLibSVMFormat.c_str();
  // read_LibSVMdata(fn,vX,vy);
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
