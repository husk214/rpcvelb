#include "Snot.h"
#include "Faop.h"

#include "L2r_huber_svc.h"
#include "Tron.h"

using namespace std;
using namespace Eigen;
using namespace sdm;

int type;
double c_l;
double c_u;
double eps;
double eps_prime;
int multi;
double alpha;
int fold_num;
bool validation_set;
int num_trials;
double inexact_level;
int numt_gp_init;
double xi;

void exit_with_help() {
  printf(
      "Computing approximation level epsilon of\n"
      " L2 regularized convex loss minimization's regularized parameter\n"
      "Usage: ./test_module [options] training_set_file1 [training_set_file2]\n"
      "                                                  "
      "[validation_set_file1]\n"
      "options:\n"
      "-s type : set type  (default 3)\n"
      "  * NOT CV\n"
      "    1 - 1) grid search \n"
      // "    1 - bisection subinterval recursively\n"
      "    2 - 2) bayesian optimization\n"
      "    3 - 3) our own method \n"
      "  * with cross validation \n"
      "    4 - 1) grid search \n"
      // "    5 - bisection subinterval recursively\n"
      "    5 - 2) bayesian optimization\n"
      "    6 - 3) our own method \n"
      // "  * finding approximately optimal parameter (FAOP)\n"
      // "    8 - using suboptimal with trick 1,2\n"
      // "    + with cross validation \n"
      // "    9 - using suboptimal with trick 1,2 \n"
      // "                                             \n"
      // "    15 - bisection subinterval recursively\n"
      // "    17 - training a midpoint of the subinterval \n"
      // "        has the lowest lower bound\n"
      "                                                     \n"
      "-l  : C_l : set  left endpoint of C interval (default 1e-3)\n"
      "-u  : C_u : set right endpoint of C interval (default 1e+3)\n"
      "                  \n"
      // " * SNOT's options\n"
      "   -n : set the number of trials (T) (default 100)\n"
      "   -i : the inexact level : set in [1e-6, 1e-1]  (default 1e-3)\n"
      "  + parameters for baysian optimization  \n"
      "   -g : the number of initial points of gaussian process in \n"
      "        baysian optimization (default 4)              \n"
      "   -x : the hyperpameter control exploration and exploitation\n"
      "        in bayesian optimization (default 0.0)\n "
      "                      \n"
      // " * FAOP's options\n"
      // "   -e : epsilon : set in (0,1) (default 0.1)\n"
      // "   -p : stopping criterion for approximate solution: set in (0, 1)\n"
      // "      (default 0.1)\n"
      // "   -m  : Trick1 parameter : set integer >= 2 (default 7)\n"
      // "   -a  : Trick2 parameter : set larger than 1 (default 1.5)\n"
      "                                                           \n"
      "-k fold : set if k-fold Cross Validation case (default 10)\n"
      "-v : flag whether or not to use validation_set (default false = -1)\n"
      "     if you use validation_set then specify -v 1\n"
      "                                            \n"
      "Output (in CV) : \n"
      "iteration epsilon C UB(E_v)*n' time(sec) ");
  std::cout << std::endl;
  exit(1);
}

void parse_command_line(int argc, char **argv, string &input_file_name,
                        string &input_file_name2) {
  int i;
  int tmp_flag;
  // default values
  type = 3;
  c_l = 1e-3;
  c_u = 1e+3;
  eps = 0.1;
  eps_prime = 0.1;
  multi = 7;
  alpha = 1.5;
  fold_num = 1;
  validation_set = false;
  num_trials = 100;
  inexact_level = 1e-3;
  numt_gp_init = 4;
  xi = 0.0;
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
    case 'n':
      num_trials = atoi(argv[i]);
      break;
    case 'i':
      inexact_level = atof(argv[i]);
      break;
    case 'g':
      numt_gp_init = atoi(argv[i]);
      break;
    case 'x':
      xi = atof(argv[i]);
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

    if ((type == 4 || type == 5 || type == 6 || type == 7 || type == 9 ||
         type == 15 || type == 17) &&
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
        equal_split_libsvm_binary(train_X, train_y, valid_X, valid_y,
                                  input_file_name);
      }
    } else {
      if (input_file_name2 == "none") {
        std::cout << "error : nothing validation_set" << std::endl;
      }
    }
    Primal_function *fun_obj = new L2r_huber_svc(train_X, train_y);
    Primal_solver *tron_obj = new Tron(fun_obj, valid_X, valid_y);
    Snot snot_obj(tron_obj, inexact_level, c_l, c_u);
    Faop faop_obj(tron_obj, eps * eps_prime, c_l, c_u);
    switch (type) {
    case 1: {
      snot_obj.with_grid_primal(num_trials);
      break;
    }
    case 0: {
      snot_obj.with_bisec_primal(num_trials);
      break;
    }
    case 2: {
      snot_obj.with_bayesian_opti_primal(num_trials, numt_gp_init, xi);
      break;
    }
    case 8: {
      faop_obj.find_app_subopt_12(multi, eps, eps * alpha);
      break;
    }
    case 99:{
      Eigen::VectorXd w_tmp;
      int num_ub, num_lb;
      tron_obj->train_warm_start_inexact(w_tmp, 1e-12, num_ub, num_lb);
      std::cout << (tron_obj->get_grad()).norm() <<std::endl;
      break;
    }
    default: { snot_obj.with_worst_lb_primal(num_trials); }
    }
  } else {
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>>
        vec_train_x, vec_valid_x;
    std::vector<Eigen::ArrayXd> vec_train_y, vec_valid_y;
    if (input_file_name2 == "none") {
      equal_split_libsvm_binary_for_cross_validation(vec_train_x, vec_train_y,
                                                     vec_valid_x, vec_valid_y,
                                                     input_file_name, fold_num);
    } else {
      merge_equal_split_libsvm_binary_for_cross_validation(
          vec_train_x, vec_train_y, vec_valid_x, vec_valid_y, input_file_name,
          input_file_name2, fold_num);
    }
    std::vector<Primal_solver *> psols;
    for (int i = 0; i < fold_num; ++i) {
      Primal_function *fun_obj_cv =
          new L2r_huber_svc((vec_train_x[i]), (vec_train_y[i]));
      Primal_solver *tron_obj_cv =
          new Tron(fun_obj_cv, (vec_valid_x[i]), (vec_valid_y[i]));
      psols.push_back(tron_obj_cv);
    }
    Snot snot_obj(psols, inexact_level, c_l, c_u);
    Faop faop_obj(psols, eps * eps_prime, c_l, c_u);
    switch (type) {
    case 4: {
      snot_obj.with_cv_grid_primal(num_trials);
      break;
    }
    // case 5: {
    //   snot_obj.with_cv_bisec_primal(num_trials);
    //   break;
    // }
    case 5: {
      snot_obj.with_cv_bayesian_opti_primal(num_trials, numt_gp_init, xi);
      break;
    }
    case 6: {
       snot_obj.with_cv_worst_lb_primal(num_trials);
       break;
    }
    case 9: {
      faop_obj.find_cv_app_subopt_12(multi, eps, eps * alpha);
      break;
    }
    case 15: {
      snot_obj.with_cv_bisec_primal(num_trials, eps);
      break;
    }
    case 17: {
      snot_obj.with_cv_worst_lb_primal(num_trials, eps);
      break;
    }
    default: { snot_obj.with_cv_worst_lb_primal(num_trials); }
    }
  }
  return 0;
}

void print_null(const char *s) {}
