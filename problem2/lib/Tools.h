#ifndef TOOLS_H_
#define TOOLS_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <numeric>
#include <iterator>
#include <list>
#include <iostream>
#include <iomanip>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
// #include <boost/algorithm/string.hpp>
// #include <boost/foreach.hpp>
#include <Eigen/Core>
#include <Eigen/SparseCore>

int string2int(const std::string &str);
double string2double(const std::string &str);

double naive(const char *p);
double naive_label(const char *p);
double naive_index(const char *p);

void read_libsvm(
    const std::string &fname_train, const std::string &fname_valid,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y);

int get_to_distribute_index(const std::vector<bool> &vec_flag,
                            const int &vec_size);

void change_vec_flag(std::vector<bool> &vec_flag, const int &vec_size,
                     const int &index);

void equal_split_libsvm_fold_cross_validation(
    const std::string &file_name, const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y);

void merge_equal_split_libsvm_fold_cross_validation(
    const std::string &file_name1, const std::string &file_name2,
    const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y);

void equal_split_libsvm(
    const std::string &fname1,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y);

void merge_equal_split_libsvm(
    const std::string &fname1, const std::string &fname2,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y);

void convert_data_format_sdm_to_libsvm(const std::string &fname_x,
                                       const std::string &fname_y,
                                       const std::string &output_fname);

std::vector<std::string> split(const std::string &str,
                               const std::string &delim);

void
read_sdm_data(const std::string &fname_x, const std::string &fname_y,
              Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
              Eigen::ArrayXd &y);

// this method has some bugs
void nearly_equal_divide_dataset(
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y);

void read_lib_rbf_basisfunc(const std::string &file_name,
                            const std::string &b_file_name, const double &sigma,
                            Eigen::MatrixXd &X, Eigen::ArrayXd &y);

void read_lib_rbf_basisfunc(
    const std::string &file_name, const std::string &b_file_name,
    const double &sigma,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y);

void read_LibSVMdata_split(
    const std::string &file_name,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> &vecX,
    std::vector<Eigen::ArrayXd> &vecy, const int &split_num);

void read_LibSVMdata_split(
    const std::string &file_name,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vecX,
    std::vector<Eigen::ArrayXd> &vecy, const int &split_num);

void read_LibSVMdata_random_split(
    const std::string &file_name,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vecX,
    std::vector<Eigen::ArrayXd> &vecy, const int &split_num);

void read_LibSVMdata_fold_cross_validation(
    const std::string &file_name, const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y);

void read_LibSVMdata_fold_cross_validation(
    const std::string &file_name, const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> &vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> &vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y);

void read_LibSVMdata1(const std::string &fname,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
                      Eigen::ArrayXd &y);

void read_LibSVMdata1(
    const std::string &fname,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y);

void read_LibSVMdata(const char *fname,
                     Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
                     Eigen::ArrayXd &y);

void read_LibSVMdata2(const std::string &fname,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
                      Eigen::VectorXd &y);
void
read_LibSVMdata(const char *fname,
                Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
                Eigen::ArrayXd &y);

void read_LibSVMdata_bias(
    const char *fname,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y);

double normsinv(const double pos);

#endif
