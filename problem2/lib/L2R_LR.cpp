#include "L2R_LR.h"

L2R_LR::L2R_LR(const std::string fileNameLibSVMFormat, const double c)
    : X(), y(), z(), D(), l(), n(), C(c) {
  read_LibSVMdata1((fileNameLibSVMFormat).c_str(), X, y);
  l = X.rows();
  n = X.cols();
}

L2R_LR::~L2R_LR() {}

L2R_LR *L2R_LR::create(const std::string fileNameLibSVMFormat) const {
  return new L2R_LR(fileNameLibSVMFormat);
}

double L2R_LR::get_func(const Eigen::VectorXd &w) {
  z = ((X * w).array());
  Eigen::ArrayXd yz = y * z;
  return (
      (w.transpose() * w)(0) / 2.0 +
      C * ((yz >= 0.0)
               .select(((-yz).exp() + 1).log(), (-yz + (yz.exp() + 1).log()))
               .sum()));
}

// -- you need using getFunc in advance, because it computes z=Xw --
Eigen::VectorXd L2R_LR::get_grad(const Eigen::VectorXd &w) {
  z = 1 / (1 + (-y * z).exp());
  D = z * (1 - z);

  return (w + X.transpose() * (C * (z - 1) * y).matrix());
}

Eigen::VectorXd L2R_LR::get_loss_grad(const Eigen::VectorXd &w) {
  z = 1 / (1 + (-y * z).exp());
  D = z * (1 - z);
  return (X.transpose() * ((z - 1) * y).matrix());
}

// compute vector producting Hessian(w) and a vector V
// Hessian(w)V = V + CX^t(D(XV))
Eigen::VectorXd L2R_LR::product_hesse_vec(const Eigen::VectorXd &V) {
  return (V + X.transpose() * (C * D * (X * V).array()).matrix());
}

int L2R_LR::get_variable() { return n; }
double L2R_LR::get_regularized_parameter(void) { return C; }
void L2R_LR::set_regularized_parameter(const double &c) { C = c; }

double L2R_LR::predict(const std::string fileNameLibSVMFormat,
                       Eigen::VectorXd &w) {
  Eigen::SparseMatrix<double, 1> vX;
  Eigen::ArrayXd vy;
  read_LibSVMdata1((fileNameLibSVMFormat).c_str(), vX, vy);
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
