#include "L2R_Huber_SVC.h"
#include "Tron.h"
#include "Validation_error_path.h"

#include <cstdio>
#include <iostream>
#include <string>
#include <cstring>
#include <fstream>

using namespace std;

int type;
double c_l;
double c_u;
double eps;
double eps_prime;
int multi;
double alpha;
int fold_num;
bool validation_set;

void exit_with_help() {
  printf(
      "Usage: ./test_module [options] training_set_file1 [training_set_file2]\n"
      "                                                  "
      "[validation_set_file1]\n"
      "options:\n"
      "-s type : set type  (default 2)\n"
      "  * Finding Approximately Regularization Parameter\n"
      "    0 - using optimal solution \n"
      "    1 - using suboptimal solution\n"
      "    2 - using suboptimal soletion + Trick1 + Trick2\n"
      "                                               \n"
      "  * Finding Approximately Regularization Parameter in k-fold CV\n"
      "    3 - op1 using optimal solution \n"
      "    4 - op2 using suboptimal\n"
      "    5 - op3 using suboptimal + Trick1 + Trick2\n"
      "                                      \n"
      "  * Finding Optimal Regularization Parameter\n"
      "    6 - optimal solution + optimal-bound \n"
      "    7 - optimal solution + optimal-bound + Trick1 \n"
      "    8 - optimal(ubeve=lbve) solution + suboptimal-bound \n"
      "    9 - optimal(ubeve=lbve) solution + suboptimal-bound + Trick1 \n"
      "                                                               \n"
      "  * Finding Optimal Regularization Parameter for fold CV\n"
      "   10 - optimal solution + optimal-bound \n"
      "   11 - optimal solution + optimal-bound + Trick1 \n"
      "                                                 \n"
      "  * Tracking Approximately Regularization Path\n"
      "   12 - optimal solution \n"
      "   13 - suboptimal solution\n"
      "                           \n"
      "  * Tracking Approximately Regularization Path in k-fold CV \n"
      "   14 - op4 optimal solution \n"
      "   15 - op5 suboptimal solution \n"
      "                           \n"
      "-l C_l : set  left endpoint of C interval (default 1e-3)\n"
      "-u C_u : set right endpoint of C interval (default 1e+3)\n"
      "-e epsilon : set in (0,1) (default 0.1)\n"
      "-p stopping criterion for approximate solution: set in (0, 1) (default 0.1)\n"
      "-m Trick1 parameter : set integer >= 2 (default 7)\n"
      "-a Trick2 parameter : set larger than 1 (default 1.5)\n"
      "-k fold : set if k-fold Cross Validation case (default 10)\n"
      "-v : flag whether or not to use validation_set (default false = -1)\n"
      "     if use validation_set then specify -v 1\n"
      "                                            \n"
      "Output (in CV) : \n"
      "  iteration trained_Reguralization_Parameter UB(Ev) E_v^{best} time(sec) \n "
      );
  std::cout << std::endl;
  exit(1);
}

void parse_command_line(int argc, char **argv, string &input_file_name,
                        string &input_file_name2) {
  int i;
  int tmp_flag;
  // default values
  type = 2;
  c_l = 1e-3;
  c_u = 1e+3;
  eps = 0.1;
  eps_prime = 0.1;
  multi = 7;
  alpha = 1.5;
  fold_num = 1;
  validation_set = false;
  // parse options
  for (i = 1; i < argc; i++) {
    if (argv[i][0] != '-')
      break;
    if (++i >= argc)
      exit_with_help();
    switch (argv[i - 1][1]) {
    case 's':
      type = atoi(argv[i]);
      break;

    case 'l':
      c_l = atof(argv[i]);
      break;
    case 'u':
      c_u = atof(argv[i]);
      break;
    case 'e':
      eps = atof(argv[i]);
      break;

    case 'p':
      eps_prime = atof(argv[i]);
      break;
    case 'm':
      multi = atoi(argv[i]);
      break;
    case 'a':
      alpha = atof(argv[i]);
      break;

    case 'k':
      fold_num = atoi(argv[i]);
      break;
    case 'v':
      tmp_flag = atoi(argv[i]);
      if (tmp_flag == 1) {
        validation_set = true;
      }
      break;
    default:
      fprintf(stderr, "unknown option: -%c\n", argv[i - 1][1]);
      exit_with_help();
      break;
    }

    if ((type == 3 || type == 4 || type == 5 || type == 10 || type == 11 ||
         type == 14 || type == 15) &&
        fold_num == 1)
      fold_num = 10;
  }

  // set_print_string_function(print_func);

  // determine filenames
  if (i >= argc)
    exit_with_help();

  input_file_name = argv[i];
  input_file_name2 = "none";
  if (i < argc - 1) {
    input_file_name2 = argv[i + 1];
  }
}

int main(int argc, char **argv) {
  string input_file_name;
  string input_file_name2;
  parse_command_line(argc, argv, input_file_name, input_file_name2);

  if (fold_num == 1) {
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> train_X,
        valid_X;
    Eigen::ArrayXd train_y, valid_y;
    if (!validation_set) {
      if (input_file_name2 == "none") {
        equal_split_libsvm(input_file_name, train_X, train_y, valid_X, valid_y);
      } else {
        merge_equal_split_libsvm(input_file_name, input_file_name2, train_X,
                                 train_y, valid_X, valid_y);
      }
    } else {
      if (input_file_name2 == "none") {
        std::cout << "error : nothing validation_set" << std::endl;
      } else {
        read_libsvm(input_file_name, input_file_name2, train_X, train_y,
                    valid_X, valid_y);
      }
    }
    Function *fun_obj = new L2R_Huber_SVC(train_X, train_y);
    Solver *tron_obj = new Tron(fun_obj, valid_X, valid_y);
    Validation_error_path vep_obj(tron_obj, valid_X, valid_y, c_l, c_u);
    switch (type) {
    case 0: {
      vep_obj.approximate_path_only_error(eps);
      break;
    }
    case 1: {
      vep_obj.approximate_inexact_train(eps, eps_prime * eps);
      break;
    }
    case 12: {
      vep_obj.approximate_path(eps);
      break;
    }
    case 13: {
      vep_obj.approximate_path_inexact(eps, eps_prime * eps);
      break;
    }
    case 6: {
      vep_obj.exact_path_only_error();
      break;
    }
    case 7: {
      vep_obj.multi_exact(multi);
      break;
    }
    case 8: {
      double train_tolerance = 1.0 / (tron_obj->get_valid_l());
      vep_obj.approximate_inexact_train(0.0, train_tolerance);
      break;
    }
    case 9: {
      double train_tolerance = 1.0 / (tron_obj->get_valid_l());
      vep_obj.apprx_inexact_train_multi_aggr(multi, 0.0, train_tolerance, 0.0);
      break;
    }
    default: {
      vep_obj.apprx_inexact_train_multi_aggr(multi, eps, eps_prime * eps,
                                             alpha * eps);
    }
    }
  } else {
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>>
        vec_train_x, vec_valid_x;
    std::vector<Eigen::ArrayXd> vec_train_y, vec_valid_y;
    if (input_file_name2 == "none") {
      equal_split_libsvm_fold_cross_validation(input_file_name, fold_num,
                                               vec_train_x, vec_train_y,
                                               vec_valid_x, vec_valid_y);
    } else {
      merge_equal_split_libsvm_fold_cross_validation(
          input_file_name, input_file_name2, fold_num, vec_train_x, vec_train_y,
          vec_valid_x, vec_valid_y);
    }
    std::vector<Solver *> v_sol;
    for (int i = 0; i < fold_num; ++i) {
      Function *fun_obj_cv =
          new L2R_Huber_SVC((vec_train_x[i]), (vec_train_y[i]));
      Solver *tron_obj_cv =
          new Tron(fun_obj_cv, (vec_valid_x[i]), (vec_valid_y[i]));
      v_sol.push_back(tron_obj_cv);
    }
    Validation_error_path vep_obj_cv(v_sol, c_l, c_u);
    switch (type) {
    case 3: {
      vep_obj_cv.cross_validation_apprx_exact(eps);
      break;
    }
    case 4: {
      vep_obj_cv.cross_validation_apprx_inexact(eps, eps_prime * eps);
      break;
    }
    case 10: {
      vep_obj_cv.cross_validation_apprx_exact(0.0);
      break;
    }
    case 11: {
      vep_obj_cv.cross_validation_apprx_exact_multi(multi, 0.0);
      break;
    }
    case 14: {
      vep_obj_cv.cross_validation_apprx_exact_path(eps);
      break;
    }
    case 15: {
      vep_obj_cv.cross_validation_apprx_inexact_path(eps, eps_prime * eps);
      break;
    }
    default: {
      vep_obj_cv.cross_validation_apprx_inexact_multi_aggr(
          multi, eps, eps_prime * eps, alpha * eps);
    }
    }
  }
  return 0;
}

void print_null(const char *s) {}
