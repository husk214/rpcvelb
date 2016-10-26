#include "Tools.h"

int string2int(const std::string &str) {
  int rt;
  std::stringstream ss;
  ss << str;
  ss >> rt;
  return rt;
}

double string2double(const std::string &str) {
  double rt;
  std::stringstream ss;
  ss << str;
  ss >> rt;
  return rt;
}

double naive(const char *p) {
  double r = 0.0;
  bool neg = false;
  if (*p == '-') {
    neg = true;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  while (*p >= '0' && *p <= '9') {
    r = (r * 10.0) + (*p - '0');
    ++p;
  }
  if (*p == '.') {
    double f = 0.0;
    int n = 0;
    ++p;
    while (*p >= '0' && *p <= '9') {
      f = (f * 10.0) + (*p - '0');
      ++p;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  if (neg) {
    r = -r;
  }
  return r;
}

double naive_label(const char *p) {
  double r = 0.0;
  bool neg = false;
  if (*p == '\n')
    ++p;
  if (*p == '-') {
    neg = true;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  while (*p >= '0' && *p <= '9') {
    r = (r * 10.0) + (*p - '0');
    ++p;
  }
  if (*p == '.') {
    double f = 0.0;
    int n = 0;
    ++p;
    while (*p >= '0' && *p <= '9') {
      f = (f * 10.0) + (*p - '0');
      ++p;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  if (neg) {
    r = -r;
  }
  return r;
}

double naive_index(const char *p) {
  double r = 0.0;
  if (*p == ' ') {
    ++p;
  }
  while (*p >= '0' && *p <= '9') {
    r = (r * 10.0) + (*p - '0');
    ++p;
  }
  if (*p == '.') {
    double f = 0.0;
    int n = 0;
    ++p;
    while (*p >= '0' && *p <= '9') {
      f = (f * 10.0) + (*p - '0');
      ++p;
      ++n;
    }
    r += f / std::pow(10.0, n);
  }
  return r;
}

void read_libsvm(
    const std::string &fname_train, const std::string &fname_valid,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y) {
  std::ifstream fs1(fname_train);
  std::ifstream fs2(fname_valid);

  if (fs1.bad() || fs1.fail() || fs2.bad() || fs2.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList1, tripletList2;
  tripletList1.reserve(1024);
  tripletList2.reserve(1024);
  train_y.resize(1024);
  valid_y.resize(1024);

  std::string buf;
  int train_n, valid_n, d;
  train_n = valid_n = d = 0;
  std::string::size_type idx1 = 0, idx2 = 0;
  double k = 0, tmp = 0;
  double label_memo = 0.0, label_tmp;
  while (std::getline(fs1, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }
    if (train_n == 0)
      label_memo = naive(buf.substr(0, idx1).c_str());

    label_tmp = naive(buf.substr(0, idx1).c_str());
    train_y.coeffRef(train_n) =  ((label_tmp == label_memo) ? 1.0 : -1.0);

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      tripletList1.push_back(T(train_n, k, tmp));

    } while (idx1 != std::string::npos);
    ++train_n;
    if (train_y.size() <= train_n) {
      train_y.conservativeResize(train_y.size() * 2);
    }
  }
  fs1.close();
  while (std::getline(fs2, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error2" << std::endl;
      std::exit(1);
    }

    label_tmp = naive(buf.substr(0, idx1).c_str());
    valid_y.coeffRef(valid_n) =  ((label_tmp == label_memo) ? 1.0 : -1.0);

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      tripletList2.push_back(T(valid_n, k, tmp));
    } while (idx1 != std::string::npos);
    ++valid_n;
    if (valid_y.size() <= valid_n) {
      valid_y.conservativeResize(valid_y.size() * 2);
    }
  }
  fs2.close();

  ++d;

  train_X.resize(train_n, d);
  train_y.conservativeResize(train_n);
  train_X.setFromTriplets(tripletList1.begin(), tripletList1.end());
  train_X.makeCompressed();

  valid_X.resize(valid_n, d);
  valid_y.conservativeResize(valid_n);
  valid_X.setFromTriplets(tripletList2.begin(), tripletList2.end());
  valid_X.makeCompressed();
}

int get_to_distribute_index(const std::vector<bool> &vec_flag,
                            const int &vec_size) {
  for (int i = 0; i < vec_size; ++i) {
    if (vec_flag[i] == true)
      return i;
  }
  std::cout << "error in get_to_distribute_index" << std::endl;
  return 0;
}

void change_vec_flag(std::vector<bool> &vec_flag, const int &vec_size,
                     const int &index) {
  vec_flag[index] = false;
  int next_index = index + 1;
  if (next_index == vec_size) {
    vec_flag[0] = true;
  } else {
    vec_flag[next_index] = true;
  }
}

void equal_split_libsvm_fold_cross_validation(
    const std::string &file_name, const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }

  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);

  typedef Eigen::Triplet<double> T;
  std::vector<double> *tmpy_array = new std::vector<double>[split_num];
  std::vector<T> *triplets_array = new std::vector<T>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::string buf;
  std::vector<int> vec_n(split_num, 0);
  std::vector<bool> vec_flag_p(split_num, false);
  std::vector<bool> vec_flag_n(split_num, false);
  int d, l, last_index, the_line_index;
  d = l = last_index = the_line_index = 0;
  double label_memo = 0.0, label_tmp = 0.0;
  vec_flag_p.at(0) = true;
  vec_flag_n.at(0) = true;
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs, buf)) {
        last_index = i;
        goto LABEL;
      }
      std::string::size_type idx1 = 0, idx2 = 0;
      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }
      if (l == 0)
        label_memo = naive(buf.substr(0, idx1).c_str());

      label_tmp = naive(buf.substr(0, idx1).c_str());
      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }

      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;
        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;
        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);
        (triplets_array[the_line_index])
            .push_back(T(vec_n[the_line_index], k,
                         naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL:
  fs.close();
  ++d;
  // ++n;
  std::vector<T> vec_x;
  std::vector<double> vec_y;

  for (int i = 0; i < split_num; ++i) {
    vec_train_x[i].resize(l - vec_n[i], d);
    vec_valid_x[i].resize(vec_n[i], d);

    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(T((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i]).setFromTriplets((triplets_array[i]).begin(),
                                     (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
}

void merge_equal_split_libsvm_fold_cross_validation(
    const std::string &file_name1, const std::string &file_name2,
    const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y) {
  std::ifstream fs1(file_name1);
  std::ifstream fs2(file_name2);

  if (fs1.bad() || fs1.fail() || fs2.bad() || fs2.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }

  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);

  typedef Eigen::Triplet<double> T;
  std::vector<double> *tmpy_array = new std::vector<double>[split_num];
  std::vector<T> *triplets_array = new std::vector<T>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::string buf;
  std::vector<int> vec_n(split_num, 0);
  std::vector<bool> vec_flag_p(split_num, false);
  std::vector<bool> vec_flag_n(split_num, false);
  int d, l, last_index, the_line_index;
  d = l = last_index = the_line_index = 0;
  double label_memo = 0.0, label_tmp = 0.0;
  vec_flag_p.at(0) = true;
  vec_flag_n.at(0) = true;
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs1, buf)) {
        last_index = i;
        goto LABEL1;
      }
      std::string::size_type idx1 = 0, idx2 = 0;
      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }
      if (l == 0)
        label_memo = naive(buf.substr(0, idx1).c_str());

      label_tmp = naive(buf.substr(0, idx1).c_str());
      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }

      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;
        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;
        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);
        (triplets_array[the_line_index])
            .push_back(T(vec_n[the_line_index], k,
                         naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL1:
  fs1.close();
  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs2, buf)) {
        last_index = i;
        goto LABEL2;
      }
      std::string::size_type idx1 = 0, idx2 = 0;
      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }

      label_tmp = naive(buf.substr(0, idx1).c_str());
      if (label_tmp == label_memo) {
        the_line_index = get_to_distribute_index(vec_flag_p, split_num);
        (tmpy_array[the_line_index]).push_back(1.0);
        change_vec_flag(vec_flag_p, split_num, the_line_index);
      } else {
        the_line_index = get_to_distribute_index(vec_flag_n, split_num);
        (tmpy_array[the_line_index]).push_back(-1.0);
        change_vec_flag(vec_flag_n, split_num, the_line_index);
      }

      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;
        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;
        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);
        (triplets_array[the_line_index])
            .push_back(T(vec_n[the_line_index], k,
                         naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      ++vec_n[the_line_index];
      ++l;
    }
  }
LABEL2:
  fs2.close();
  ++d;
  // ++n;
  std::vector<T> vec_x;
  std::vector<double> vec_y;

  for (int i = 0; i < split_num; ++i) {
    vec_train_x[i].resize(l - vec_n[i], d);
    vec_valid_x[i].resize(vec_n[i], d);

    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(T((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i]).setFromTriplets((triplets_array[i]).begin(),
                                     (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
}

void equal_split_libsvm(
    const std::string &fname1,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y) {
  std::ifstream fs1(fname1);

  if (fs1.bad() || fs1.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList1, tripletList2;
  tripletList1.reserve(1024);
  tripletList2.reserve(1024);
  train_y.resize(1024);
  valid_y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;
  std::string::size_type idx1 = 0, idx2 = 0;
  double k = 0, tmp = 0;
  double label_memo = 0.0, label_tmp;
  bool flag_p = true, flag_n = true, flag_x = true;
  while (std::getline(fs1, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }
    if (n == 0)
      label_memo = naive(buf.substr(0, idx1).c_str());

    label_tmp = naive(buf.substr(0, idx1).c_str());
    if (label_tmp == label_memo) {
      if (flag_p) {
        train_y.coeffRef(n) = 1.0;
        flag_p = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = 1.0;
        flag_p = true;
        flag_x = false;
      }
    } else {
      if (flag_n) {
        train_y.coeffRef(n) = -1.0;
        flag_n = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = -1.0;
        flag_n = true;
        flag_x = false;
      }
    }

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      if (flag_x) {
        tripletList1.push_back(T(n, k, tmp));
      } else {
        tripletList2.push_back(T(n, k, tmp));
      }
    } while (idx1 != std::string::npos);
    if (!flag_x)
      ++n;
    if (train_y.size() <= n) {
      train_y.conservativeResize(train_y.size() * 2);
      valid_y.conservativeResize(valid_y.size() * 2);
    }
  }
  fs1.close();

  ++d;
  valid_y.conservativeResize(n);
  if (flag_x) {
    train_X.resize(n + 1, d);
    train_y.conservativeResize(n + 1);
  } else {
    train_X.resize(n, d);
    train_y.conservativeResize(n);
  }
  train_X.setFromTriplets(tripletList1.begin(), tripletList1.end());
  train_X.makeCompressed();
  valid_X.resize(n, d);
  valid_X.setFromTriplets(tripletList2.begin(), tripletList2.end());
  valid_X.makeCompressed();
}

void merge_equal_split_libsvm(
    const std::string &fname1, const std::string &fname2,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y) {
  std::ifstream fs1(fname1);
  std::ifstream fs2(fname2);

  if (fs1.bad() || fs1.fail() || fs2.bad() || fs2.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList1, tripletList2;
  tripletList1.reserve(1024);
  tripletList2.reserve(1024);
  train_y.resize(1024);
  valid_y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;
  std::string::size_type idx1 = 0, idx2 = 0;
  double k = 0, tmp = 0;
  double label_memo = 0.0, label_tmp;
  bool flag_p = true, flag_n = true, flag_x = true;
  while (std::getline(fs1, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }
    if (n == 0)
      label_memo = naive(buf.substr(0, idx1).c_str());

    label_tmp = naive(buf.substr(0, idx1).c_str());
    if (label_tmp == label_memo) {
      if (flag_p) {
        train_y.coeffRef(n) = 1.0;
        flag_p = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = 1.0;
        flag_p = true;
        flag_x = false;
      }
    } else {
      if (flag_n) {
        train_y.coeffRef(n) = -1.0;
        flag_n = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = -1.0;
        flag_n = true;
        flag_x = false;
      }
    }

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      if (flag_x) {
        tripletList1.push_back(T(n, k, tmp));
      } else {
        tripletList2.push_back(T(n, k, tmp));
      }
    } while (idx1 != std::string::npos);
    if (!flag_x)
      ++n;
    if (train_y.size() <= n) {
      train_y.conservativeResize(train_y.size() * 2);
      valid_y.conservativeResize(valid_y.size() * 2);
    }
  }
  fs1.close();
  while (std::getline(fs2, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }

    label_tmp = naive(buf.substr(0, idx1).c_str());
    if (label_tmp == label_memo) {
      if (flag_p) {
        train_y.coeffRef(n) = 1.0;
        flag_p = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = 1.0;
        flag_p = true;
        flag_x = false;
      }
    } else {
      if (flag_n) {
        train_y.coeffRef(n) = -1.0;
        flag_n = false;
        flag_x = true;
      } else {
        valid_y.coeffRef(n) = -1.0;
        flag_n = true;
        flag_x = false;
      }
    }

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      if (flag_x) {
        tripletList1.push_back(T(n, k, tmp));
      } else {
        tripletList2.push_back(T(n, k, tmp));
      }
    } while (idx1 != std::string::npos);
    if (!flag_x)
      ++n;
    if (train_y.size() <= n) {
      train_y.conservativeResize(train_y.size() * 2);
      valid_y.conservativeResize(valid_y.size() * 2);
    }
  }
  fs2.close();

  ++d;
  valid_y.conservativeResize(n);
  if (flag_x) {
    train_X.resize(n + 1, d);
    train_y.conservativeResize(n + 1);
  } else {
    train_X.resize(n, d);
    train_y.conservativeResize(n);
  }
  train_X.setFromTriplets(tripletList1.begin(), tripletList1.end());
  train_X.makeCompressed();
  valid_X.resize(n, d);
  valid_X.setFromTriplets(tripletList2.begin(), tripletList2.end());
  valid_X.makeCompressed();
}

void convert_data_format_sdm_to_libsvm(const std::string &fname_x,
                                       const std::string &fname_y,
                                       const std::string &output_fname) {
  std::ifstream fs(fname_x);
  std::ifstream label_fs(fname_y);
  if (fs.bad() || fs.fail() || label_fs.bad() || label_fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  std::string buf, label_buf, tmp_st1, tmp_st2;
  std::getline(fs, buf);
  std::getline(label_fs, label_buf);
  std::vector<std::string> vec1, vec2;
  vec1 = split(buf, " ");
  vec2 = split(label_buf, " ");
  if (vec1.at(0) != "#" || vec2.at(0) != "#" || vec1.size() != 3 ||
      vec2.size() != 2) {
    std::cout << "file's style error" << std::endl;
    std::exit(1);
  }
  if (vec1.at(1) != vec2.at(1)) {
    std::cout << "number of dataset and label instance do not much"
              << std::endl;
    std::exit(1);
  }

  std::ofstream ofs(output_fname);
  std::string for_out, tmp_st;

  std::string::size_type idx1 = 0, idx2 = 0;
  int k = 1;
  while (std::getline(fs, buf)) {
    idx1 = 0, idx2 = 0;
    k = 1;
    std::getline(label_fs, label_buf);
    for_out = std::to_string((naive_label(label_buf.c_str()) > 0.0) ? 1 : -1);
    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(" \t", idx1)) == std::string::npos) {
        if ((tmp_st = (buf.substr(idx1))) != "0")
          for_out += " " + std::to_string(k) + ":" + tmp_st;
        break;
      } else {
        if ((tmp_st = (buf.substr(idx1, idx2 - idx1))) != "0")
          for_out += " " + std::to_string(k) + ":" + tmp_st;
      }
      ++idx2;
      idx1 = idx2;
      ++k;
    } while (idx1 != std::string::npos);
    ofs << for_out << std::endl;
  }
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  std::string::size_type current = 0, found, delimlen = delim.size();
  while ((found = str.find(delim, current)) != std::string::npos) {
    res.push_back(std::string(str, current, found - current));
    current = found + delimlen;
  }
  res.push_back(std::string(str, current, str.size() - current));
  return res;
}

void
read_sdm_data(const std::string &fname_x, const std::string &fname_y,
              Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
              Eigen::ArrayXd &y) {
  std::ifstream fs(fname_x);
  std::ifstream label_fs(fname_y);
  if (fs.bad() || fs.fail() || label_fs.bad() || label_fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  std::string buf, label_buf, tmp_st1, tmp_st2;
  std::getline(fs, buf);
  std::getline(label_fs, label_buf);
  std::vector<std::string> vec1, vec2;
  vec1 = split(buf, " ");
  vec2 = split(label_buf, " ");
  if (vec1.at(0) != "#" || vec2.at(0) != "#" || vec1.size() != 3 ||
      vec2.size() != 2) {
    std::cout << "file's style error" << std::endl;
    std::exit(1);
  }
  if (vec1.at(1) != vec2.at(1)) {
    std::cout << "number of dataset and label instance do not much"
              << std::endl;
    std::exit(1);
  }
  int x_rows = string2int(vec1.at(1));
  int x_cols = string2int(vec1.at(2));

  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(x_rows * x_cols);
  y.resize(x_rows);

  int n, d;
  n = d = 0;
  std::string::size_type idx1 = 0, idx2 = 0;
  double k = 0, tmp = 0;
  while (std::getline(fs, buf)) {
    idx1 = 0, idx2 = 0;
    k = 0;
    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(" \t", idx1)) == std::string::npos) {
        tmp = naive((buf.substr(idx1)).c_str());
        tripletList.push_back(T(n, k, tmp));
        break;
      } else {
        tmp = naive((buf.substr(idx1, idx2 - idx1)).c_str());
      }

      ++idx2;
      idx1 = idx2;
      tripletList.push_back(T(n, k, tmp));
      ++k;
    } while (idx1 != std::string::npos);
    std::getline(label_fs, label_buf);
    y.coeffRef(n++) = ((naive_label(label_buf.c_str()) > 0.0) ? 1.0 : -1.0);
    // if (y.size() <= (++n))
    //   y.conservativeResize(y.size() * 2);
  }
  fs.close();
  ++d;
  // y.conservativeResize();
  X.resize(x_rows, x_cols);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  X.makeCompressed();
}

// this method has some bugs
void nearly_equal_divide_dataset(
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &train_X,
    Eigen::ArrayXd &train_y,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &valid_X,
    Eigen::ArrayXd &valid_y) {
  int num_instance = X.rows();
  int x_cols = X.cols();
  std::vector<int> p_label_index, n_label_index;
  for (int i = 0; i < num_instance; ++i) {
    if (y.coeffRef(i) == 1.0) {
      p_label_index.push_back(i);
    } else {
      n_label_index.push_back(i);
    }
  }
  int half_num_ins = num_instance / 2;
  train_X.resize(half_num_ins + (num_instance % 2), x_cols);
  train_y.resize(half_num_ins + (num_instance % 2));
  valid_X.resize(half_num_ins, x_cols);
  valid_y.resize(half_num_ins);

  int count_t = 0, count_v = 0, index = 0, num_p = p_label_index.size() - 1;
  while (1) {
    if (index > num_p)
      break;
    train_X.row(count_t) = X.row(p_label_index[index]);
    train_y.coeffRef(count_t++) = y.coeffRef(p_label_index[index++]);
    if (index > num_p)
      break;
    valid_X.row(count_v) = X.row(p_label_index[index]);
    valid_y.coeffRef(count_v++) = y.coeffRef(p_label_index[index++]);
  }
  index = 0;
  int num_n = n_label_index.size() - 1;
  while (1) {
    if (index > num_n)
      break;
    train_X.row(count_t) = X.row(n_label_index[index]);
    train_y.coeffRef(count_t++) = y.coeffRef(n_label_index[index++]);
    if (index > num_n)
      break;
    valid_X.row(count_v) = X.row(n_label_index[index]);
    valid_y.coeffRef(count_v++) = y.coeffRef(n_label_index[index++]);
  }
}

void read_lib_rbf_basisfunc(const std::string &file_name,
                            const std::string &b_file_name, const double &sigma,
                            Eigen::MatrixXd &X, Eigen::ArrayXd &y) {
  Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> sX;
  read_LibSVMdata1(file_name, sX, y);
  Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> basicX;
  Eigen::ArrayXd basicy;
  read_LibSVMdata1(b_file_name, basicX, basicy);
  int l = sX.rows();
  int n = basicX.rows();
  X.resize(l, n);
  Eigen::SparseVector<double> xi, xj;
  double sigma2 = 2.0 * sigma * sigma;
  for (int i = 0; i < l; ++i) {
    xi = sX.row(i);
    for (int j = 0; j < n; ++j) {
      xj = basicX.row(j);
      X.coeffRef(i, j) = exp(-((xi - xj).squaredNorm()) / sigma2);
    }
  }
}

void read_lib_rbf_basisfunc(
    const std::string &file_name, const std::string &b_file_name,
    const double &sigma,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y) {

  read_LibSVMdata1(file_name, X, y);
  Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> basicX;
  Eigen::ArrayXd basicy;
  read_LibSVMdata1(b_file_name, basicX, basicy);

  int l = X.rows();
  int n = basicX.rows();
  Eigen::MatrixXd bX(l, n);
  Eigen::SparseVector<double> xi, xj;
  double sigma2 = 2.0 * sigma * sigma;
  for (int i = 0; i < l; ++i) {
    xi = X.row(i);
    for (int j = 0; j < n; ++j) {
      xj = basicX.row(j);
      bX.coeffRef(i, j) = exp(-((xi - xj).squaredNorm()) / sigma2);
    }
  }
  X = bX.sparseView();
}

void read_LibSVMdata_split(
    const std::string &file_name,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> &vecX,
    std::vector<Eigen::ArrayXd> &vecy, const int &split_num) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }

  vecX.resize(split_num);
  vecy.resize(split_num);
  typedef Eigen::Triplet<double> T;
  std::vector<double> *tmpy_array = new std::vector<double>[split_num];
  std::vector<T> *triplets_array = new std::vector<T>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::string buf;
  int n, d, last_index = 0;
  n = d = 0;

  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs, buf)) {

        last_index = i;
        goto LABEL;
      }
      std::string::size_type idx1 = 0, idx2 = 0;

      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }
      (tmpy_array[i]).push_back(
          (naive_label(buf.substr(0, idx1).c_str()) > 0.0) ? 1.0 : -1.0);
      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;

        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;

        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);

        (triplets_array[i])
            .push_back(T(n, k, naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      if (i == (split_num - 1))
        ++n;
    }
  }
LABEL:
  fs.close();
  ++d;
  // ++n;
  for (int i = 0; i < split_num; ++i) {
    if (i >= last_index) {
      (vecX[i]).resize(n, d);
    } else {
      (vecX[i]).resize(n + 1, d);
    }

    (vecX[i]).setFromTriplets((triplets_array[i]).begin(),
                              (triplets_array[i]).end());

    (vecX[i]).makeCompressed();
    (vecy[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
}

void read_LibSVMdata_split(
    const std::string &file_name,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vecX,
    std::vector<Eigen::ArrayXd> &vecy, const int &split_num) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }

  vecX.resize(split_num);
  vecy.resize(split_num);
  typedef Eigen::Triplet<double> T;
  std::vector<double> *tmpy_array = new std::vector<double>[split_num];
  std::vector<T> *triplets_array = new std::vector<T>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::string buf;
  int n, d, last_index = 0;
  n = d = 0;

  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs, buf)) {

        last_index = i;
        goto LABEL;
      }
      std::string::size_type idx1 = 0, idx2 = 0;

      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }
      (tmpy_array[i]).push_back(
          (naive_label(buf.substr(0, idx1).c_str()) > 0.0) ? 1.0 : -1.0);
      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;

        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;

        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);

        (triplets_array[i])
            .push_back(T(n, k, naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      if (i == (split_num - 1))
        ++n;
    }
  }
LABEL:
  fs.close();
  ++d;
  // ++n;
  for (int i = 0; i < split_num; ++i) {
    if (i >= last_index) {
      (vecX[i]).resize(n, d);
    } else {
      (vecX[i]).resize(n + 1, d);
    }

    (vecX[i]).setFromTriplets((triplets_array[i]).begin(),
                              (triplets_array[i]).end());

    (vecX[i]).makeCompressed();
    (vecy[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
}

void read_LibSVMdata_random_split(
    const std::string &file_name,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vecX,
    std::vector<Eigen::ArrayXd> &vecy, const int &split_num) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  std::mt19937 engine((static_cast<unsigned long>(time(nullptr))));

  std::vector<int> index_determin(split_num);
  std::iota(index_determin.begin(), index_determin.end(), 0);

  std::vector<int> index_count(split_num, 0);
  vecX.resize(split_num);
  vecy.resize(split_num);
  typedef Eigen::Triplet<double> T;
  std::vector<double> *tmpy_array = new std::vector<double>[split_num];
  std::vector<T> *triplets_array = new std::vector<T>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::string buf;
  int n, d;
  n = d = 0;

  while (1) {
    std::shuffle(index_determin.begin(), index_determin.end(),
                 std::mt19937(engine));
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs, buf)) {
        goto LABEL;
      }
      ++index_count[index_determin[i]];
      std::string::size_type idx1 = 0, idx2 = 0;

      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }
      (tmpy_array[index_determin[i]]).push_back(
          (naive_label(buf.substr(0, idx1).c_str()) > 0.0) ? 1.0 : -1.0);
      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;

        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;

        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);

        (triplets_array[index_determin[i]])
            .push_back(T(n, k, naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      if (i == (split_num - 1))
        ++n;
    }
  }
LABEL:
  fs.close();
  ++d;
  // ++n;
  for (int i = 0; i < split_num; ++i) {
    (vecX[i]).resize(index_count[i], d);

    (vecX[i]).setFromTriplets((triplets_array[i]).begin(),
                              (triplets_array[i]).end());

    (vecX[i]).makeCompressed();
    (vecy[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
}

void read_LibSVMdata_fold_cross_validation(
    const std::string &file_name, const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t>> &
        vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }

  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);

  typedef Eigen::Triplet<double> T;
  std::vector<double> *tmpy_array = new std::vector<double>[split_num];
  std::vector<T> *triplets_array = new std::vector<T>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::string buf;
  int n, d, l, last_index;
  n = d = l = last_index = 0;

  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs, buf)) {
        last_index = i;
        goto LABEL;
      }
      ++l;
      std::string::size_type idx1 = 0, idx2 = 0;
      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }
      (tmpy_array[i]).push_back(
          (naive_label(buf.substr(0, idx1).c_str()) > 0.0) ? 1.0 : -1.0);
      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;
        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;
        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);
        (triplets_array[i])
            .push_back(T(n, k, naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      if (i == (split_num - 1))
        ++n;
    }
  }
LABEL:
  fs.close();
  ++d;
  // ++n;
  std::vector<T> vec_x;
  std::vector<double> vec_y;

  for (int i = 0; i < split_num; ++i) {
    if (i >= last_index) {
      (vec_train_x[i]).resize(l - n, d);
      (vec_valid_x[i]).resize(n, d);
    } else {
      (vec_train_x[i]).resize(l - n - 1, d);
      (vec_valid_x[i]).resize(n + 1, d);
    }
    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(T((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i]).setFromTriplets((triplets_array[i]).begin(),
                                     (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
}

void read_LibSVMdata_fold_cross_validation(
    const std::string &file_name, const int &split_num,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> &vec_train_x,
    std::vector<Eigen::ArrayXd> &vec_train_y,
    std::vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> &vec_valid_x,
    std::vector<Eigen::ArrayXd> &vec_valid_y) {
  std::ifstream fs(file_name);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }

  vec_train_x.resize(split_num);
  vec_train_y.resize(split_num);
  vec_valid_x.resize(split_num);
  vec_valid_y.resize(split_num);

  typedef Eigen::Triplet<double> T;
  std::vector<double> *tmpy_array = new std::vector<double>[split_num];
  std::vector<T> *triplets_array = new std::vector<T>[split_num];
  for (int i = 0; i < split_num; ++i) {
    (triplets_array[i]).reserve(1024);
    (tmpy_array[i]).reserve(1024);
  }

  std::string buf;
  int n, d, l, last_index;
  n = d = l = last_index = 0;

  while (1) {
    for (int i = 0; i < split_num; ++i) {
      if (!std::getline(fs, buf)) {
        last_index = i;
        goto LABEL;
      }
      ++l;
      std::string::size_type idx1 = 0, idx2 = 0;
      if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
        std::cout << "file format error1" << std::endl;
        std::exit(1);
      }
      (tmpy_array[i]).push_back(
          (naive_label(buf.substr(0, idx1).c_str()) > 0.0) ? 1.0 : -1.0);
      do {
        --idx1;
        ++idx1;
        if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
          break;
        int k = naive_index(buf.substr(idx1, idx2 - idx1).c_str()) - 1;
        if (d < k)
          d = k;
        ++idx2;
        idx1 = buf.find_first_of(" \t", idx2);
        (triplets_array[i])
            .push_back(T(n, k, naive(buf.substr(idx2, idx1 - idx2).c_str())));
      } while (idx1 != std::string::npos);
      if (i == (split_num - 1))
        ++n;
    }
  }
LABEL:
  fs.close();
  ++d;
  // ++n;
  std::vector<T> vec_x;
  std::vector<double> vec_y;

  for (int i = 0; i < split_num; ++i) {
    if (i >= last_index) {
      (vec_train_x[i]).resize(l - n, d);
      (vec_valid_x[i]).resize(n, d);
    } else {
      (vec_train_x[i]).resize(l - n - 1, d);
      (vec_valid_x[i]).resize(n + 1, d);
    }
    vec_x.clear();
    vec_y.clear();
    int count_cv = 0;
    for (int j = 0; j < split_num; ++j) {
      if (j != i) {
        if (count_cv == 0) {
          vec_x.insert(vec_x.end(), (triplets_array[j]).begin(),
                       (triplets_array[j]).end());
        } else {
          for (auto it : (triplets_array[j]))
            vec_x.push_back(T((it.row() + count_cv), it.col(), it.value()));
        }
        vec_y.insert(vec_y.end(), (tmpy_array[j]).begin(),
                     (tmpy_array[j]).end());
        count_cv += (tmpy_array[j]).size();
      }
    }
    (vec_train_x[i]).setFromTriplets(vec_x.begin(), vec_x.end());
    (vec_valid_x[i]).setFromTriplets((triplets_array[i]).begin(),
                                     (triplets_array[i]).end());

    (vec_train_x[i]).makeCompressed();
    (vec_valid_x[i]).makeCompressed();

    (vec_train_y[i]) = Eigen::Map<Eigen::ArrayXd>(&(vec_y)[0], (vec_y).size());
    (vec_valid_y[i]) =
        Eigen::Map<Eigen::ArrayXd>(&(tmpy_array[i])[0], (tmpy_array[i]).size());
  }
  delete[] triplets_array;
  delete[] tmpy_array;
}

void read_LibSVMdata1(const std::string &fname,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
                      Eigen::ArrayXd &y) {
  std::ifstream fs(fname);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;
  std::string::size_type idx1 = 0, idx2 = 0;
  double k = 0, tmp = 0;
  double label_memo = 0.0;
  while (std::getline(fs, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }
    if (n == 0) {
      label_memo = naive(buf.substr(0, idx1).c_str());
    }
    y.coeffRef(n) =
        (naive(buf.substr(0, idx1).c_str()) == label_memo) ? 1.0 : -1.0;
    std::cout << label_memo << " " << naive(buf.substr(0, idx1).c_str())
              << std::endl;
    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      tripletList.push_back(T(n, k, tmp));
    } while (idx1 != std::string::npos);

    if (y.size() <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  fs.close();
  ++d;
  y.conservativeResize(n);
  X.resize(n, d);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  X.makeCompressed();
}

void read_LibSVMdata1(
    const std::string &fname,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y) {
  std::ifstream fs(fname);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;
  std::string::size_type idx1 = 0, idx2 = 0;
  double k = 0, tmp = 0;
  double label_memo = 0.0, tmp_label = 0.0;
  bool label_flag = false;
  while (std::getline(fs, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }
    tmp_label = naive(buf.substr(0, idx1).c_str());
    if (n == 0)
      label_memo = tmp_label;
    if (label_flag) {
      y.coeffRef(n) = (tmp_label == label_memo) ? 1.0 : -1.0;
    } else {
      if (label_memo == tmp_label) {
        y.coeffRef(n) = 1.0;
      } else {
        if (label_memo > tmp_label) {
          y.coeffRef(n) = -1.0;
        } else {
          for (int i = 0; i < n; ++i)
            y.coeffRef(i) = -1.0;
          y.coeffRef(n) = 1.0;
          label_memo = tmp_label;
          label_flag = true;
        }
      }
    }
    // y.coeffRef(n) =
    //     (naive(buf.substr(0, idx1).c_str()) == label_memo) ? 1.0 : -1.0;
    // std::cout << label_memo << " " << naive(buf.substr(0, idx1).c_str()) << "
    // "
    //           << y.coeffRef(n) << std::endl;

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      tripletList.push_back(T(n, k, tmp));
    } while (idx1 != std::string::npos);

    if (y.size() <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  fs.close();
  ++d;
  y.conservativeResize(n);
  X.resize(n, d);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  X.makeCompressed();
}

void read_LibSVMdata(const char *fname,
                     Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
                     Eigen::ArrayXd &y) {
  std::ifstream fs(fname);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;

  while (std::getline(fs, buf)) {
    std::string::size_type idx1 = 0, idx2 = 0;

    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }

    // use a C code "strtod/strtod", because of C++ "sstream" is slow.
    // In addition, if you use strtod and std::string, you have to use c_str()
    // is too slow,
    // so you should not use strtod together with std::string.
    y[n] = strtod(buf.substr(0, idx1).c_str(), NULL);
    y[n] = (y[n] > 0.0) ? 1.0 : -1.0;

    // tripletList.push_back(T(n, 0, 1.0)); // bias

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      int k = strtod(buf.substr(idx1, idx2 - idx1).c_str(), NULL);
      if (d < k)
        d = k;

      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      double tmp = strtod(buf.substr(idx2, idx1 - idx2).c_str(), NULL);
      tripletList.push_back(T(n, k, tmp));
    } while (idx1 != std::string::npos);

    if (y.size() <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  d += 1; // for bias
  fs.close();

  y.conservativeResize(n);
  X.resize(n, d);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  X.makeCompressed();
}

void read_LibSVMdata2(const std::string &fname,
                      Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
                      Eigen::VectorXd &y) {
  std::ifstream fs(fname);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;
  std::string::size_type idx1 = 0, idx2 = 0;
  double k = 0, tmp = 0;
  while (std::getline(fs, buf)) {
    idx1 = 0, idx2 = 0;
    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }
    y.coeffRef(n) = (naive(buf.substr(0, idx1).c_str()) > 0.0) ? 1.0 : -1.0;

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      k = naive_index((buf.substr(idx1, idx2 - idx1)).c_str()) - 1;
      if (d < k)
        d = k;
      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      tmp = naive((buf.substr(idx2, idx1 - idx2)).c_str());
      tripletList.push_back(T(n, k, tmp));
    } while (idx1 != std::string::npos);

    if (y.size() <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  fs.close();
  ++d;
  y.conservativeResize(n);
  X.resize(n, d);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  X.makeCompressed();
}

void
read_LibSVMdata(const char *fname,
                Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
                Eigen::ArrayXd &y) {
  std::ifstream fs(fname);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;

  while (std::getline(fs, buf)) {
    std::string::size_type idx1 = 0, idx2 = 0;

    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }

    // use a C code "strtod/strtod", because of C++ "sstream" is slow.
    // In addition, if you use strtod and std::string, you have to use c_str()
    // is too slow,
    // so you should not use strtod together with std::string.
    y[n] = strtod(buf.substr(0, idx1).c_str(), NULL);
    y[n] = (y[n] > 0.0) ? 1.0 : -1.0;

    // tripletList.push_back(T(n, 0, 1.0)); // bias

    do {
      --idx1;
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      int k = strtod(buf.substr(idx1, idx2 - idx1).c_str(), NULL);
      if (d < k)
        d = k;

      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      double tmp = strtod(buf.substr(idx2, idx1 - idx2).c_str(), NULL);
      tripletList.push_back(T(n, k, tmp));
    } while (idx1 != std::string::npos);

    if (y.size() <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  d += 1; // for bias
  fs.close();

  y.conservativeResize(n);
  X.resize(n, d);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  X.makeCompressed();
}

void read_LibSVMdata_bias(
    const char *fname,
    Eigen::SparseMatrix<double, Eigen::RowMajor, std::ptrdiff_t> &X,
    Eigen::ArrayXd &y) {
  std::ifstream fs(fname);
  if (fs.bad() || fs.fail()) {
    std::cout << "file open error" << std::endl;
    std::exit(1);
  }
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList;
  tripletList.reserve(1024); // estimation of non_zero_entries
  y.resize(1024);

  std::string buf;
  int n, d;
  n = d = 0;

  while (std::getline(fs, buf)) {
    std::string::size_type idx1 = 0, idx2 = 0;

    if ((idx1 = buf.find_first_of(" \t", 0)) == std::string::npos) {
      std::cout << "file format error1" << std::endl;
      std::exit(1);
    }

    // use a C code "strtod/strtod", because of C++ "sstream" is slow.
    // In addition, if you use strtod and std::string, you have to use c_str()
    // is too slow,
    // so you should not use strtod together with std::string.
    y[n] = strtod(buf.substr(0, idx1).c_str(), NULL);
    y[n] = (y[n] > 0.0) ? 1.0 : -1.0;

    tripletList.push_back(T(n, 0, 1.0)); // bias

    do {
      ++idx1;
      if ((idx2 = buf.find_first_of(":", idx1)) == std::string::npos)
        break;

      int k = strtod(buf.substr(idx1, idx2 - idx1).c_str(), NULL);
      if (d < k)
        d = k;

      ++idx2;
      idx1 = buf.find_first_of(" \t", idx2);

      double tmp = strtod(buf.substr(idx2, idx1 - idx2).c_str(), NULL);

      tripletList.push_back(T(n, k, tmp));
    } while (idx1 != std::string::npos);

    if (y.size() <= (++n))
      y.conservativeResize(y.size() * 2);
  }
  d += 1; // for bias
  fs.close();

  y.conservativeResize(n);
  X.resize(n, d);
  X.setFromTriplets(tripletList.begin(), tripletList.end());
  X.makeCompressed();
}

double normsinv(const double pos) {

  const double A1 = (-3.969683028665376e+01);
  const double A2 = 2.209460984245205e+02;
  const double A3 = (-2.759285104469687e+02);
  const double A4 = 1.383577518672690e+02;
  const double A5 = (-3.066479806614716e+01);
  const double A6 = 2.506628277459239e+00;

  const double B1 = (-5.447609879822406e+01);
  const double B2 = 1.615858368580409e+02;
  const double B3 = (-1.556989798598866e+02);
  const double B4 = 6.680131188771972e+01;
  const double B5 = (-1.328068155288572e+01);

  const double C1 = (-7.784894002430293e-03);
  const double C2 = (-3.223964580411365e-01);
  const double C3 = (-2.400758277161838e+00);
  const double C4 = (-2.549732539343734e+00);
  const double C5 = 4.374664141464968e+00;
  const double C6 = 2.938163982698783e+00;

  const double D1 = 7.784695709041462e-03;
  const double D2 = 3.224671290700398e-01;
  const double D3 = 2.445134137142996e+00;
  const double D4 = 3.754408661907416e+00;

  const double P_LOW = 0.02425;
  /* P_high = 1 - p_low*/
  const double P_HIGH = 0.97575;

  double x = 0.0;
  double q, r;
  if ((0 < pos) && (pos < P_LOW)) {
    q = sqrt(-2 * log(pos));
    x = (((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) /
        ((((D1 * q + D2) * q + D3) * q + D4) * q + 1);
  } else {
    if ((P_LOW <= pos) && (pos <= P_HIGH)) {
      q = pos - 0.5;
      r = q * q;
      x = (((((A1 * r + A2) * r + A3) * r + A4) * r + A5) * r + A6) * q /
          (((((B1 * r + B2) * r + B3) * r + B4) * r + B5) * r + 1);
    } else {
      if ((P_HIGH < pos) && (pos < 1)) {
        q = sqrt(-2 * log(1 - pos));
        x = -(((((C1 * q + C2) * q + C3) * q + C4) * q + C5) * q + C6) /
            ((((D1 * q + D2) * q + D3) * q + D4) * q + 1);
      }
    }
  }

  /* If you are composiling this under UNIX OR LINUX, you may uncomment this
  block for better accuracy.
  if(( 0 < pos)&&(pos < 1)){
    e = 0.5 * erfc(-x/sqrt(2)) - pos;
    u = e * sqrt(2*M_PI) * exp(x*x/2);
    x = x - u/(1 + x*u/2);
  }
  */

  return x;
}

// void  loadLib(const std::string& filename, Eigen::SparseMatrix<double,
// Eigen::RowMajor, std::ptrdiff_t>& X,Eigen::ArrayXd& y ){
//   int colLen = 0, maxIndex = 0, tmpIndex , nnz=0, num_posi=0, num_nega=0;
//   double yValue, yValueType1 = -1.0, yValueType2 = 1.0;
//   bool flag1 = false , flag2 = false;
//   std::vector<double> tmpy;
//   typedef Eigen::Triplet<double> T;
//   std::vector<T> v;

//   std::string buff;
//   std::ifstream ifs(filename);
//   std::list<std::string> results;
//   std::list<std::string> results1;
//   if (ifs.fail())
//   {
//     std::cerr << "can't open input file " << filename <<std::endl;
//     std::exit(1);
//   }

//   while (getline(ifs, buff))
//   {
//     boost::split(results, buff,  boost::is_space());
//     yValue = string2double(results.front());
//     if(!flag1){
//       yValueType1 = yValue;
//       flag1 = true;
//     }
//     else if(!flag2 && yValue != yValueType1){
//       yValueType2 = yValue;
//       flag2 = true;
//     }
//     if(flag1 && flag2) break;
//   }
//   ifs.close();
//   ifs.open(filename);

//   if(yValueType1 > yValueType2){
//     double tmp = yValueType2;
//     yValueType2 = yValueType1;
//     yValueType1 = tmp;
//   }
//   // std::cout << yValueType1 <<" " <<yValueType2 <<std::endl;
//   if(yValueType1 == -1.0 && yValueType2 == 1.0){
//     while (getline(ifs, buff))
//     {
//       boost::split(results, buff,  boost::is_space());
//       yValue = string2double(results.front());
//       tmpy.push_back(yValue);
//       results.pop_front();
//       if(yValue == -1.0) {
//         ++num_nega;
//       }
//       else {
//         ++num_posi;
//       }
//       BOOST_FOREACH(std::string p, results) {
//         tmpIndex = 0;
//         if(p.find_first_of(":") != std::string::npos){
//           ++nnz;
//           boost::split(results1, p,  boost::is_any_of(":"));
//           tmpIndex = string2int(results1.front());
//           results1.pop_front();
//           v.push_back(T(colLen, tmpIndex-1, string2double(results1.front())
//           ));
//         }
//         if(maxIndex < tmpIndex) maxIndex = tmpIndex;
//       }
//       ++colLen;
//     }
//   }
//   else{
//     while (getline(ifs, buff))
//     {
//       boost::split(results, buff,  boost::is_space());
//       yValue = string2double(results.front());
//       if(yValue == yValueType1) {
//         tmpy.push_back(-1.0);
//         ++num_nega;
//       }
//       else {
//         tmpy.push_back(1.0);
//         ++num_posi;
//       }
//       results.pop_front();

//       BOOST_FOREACH(std::string p, results) {
//         tmpIndex = 0;
//         if(p.find_first_of(":") != std::string::npos){
//           ++nnz;
//           boost::split(results1, p,  boost::is_any_of(":"));
//           tmpIndex = string2int(results1.front());
//           results1.pop_front();
//           v.push_back(T(colLen, tmpIndex-1, string2double(results1.front())
//           ));
//         }
//         if(maxIndex < tmpIndex) maxIndex = tmpIndex;
//       }
//       ++colLen;
//     }
//   }
//   std::cout <<"#data"<< colLen <<" #Posi " <<num_posi <<" #Nega " <<num_nega
//   <<" #features" <<maxIndex <<" #nz " << nnz <<std::endl;
//   X.resize(colLen, maxIndex);

//   X.setFromTriplets(v.begin(), v.end());
//   X.makeCompressed();
//   y = Eigen::Map<Eigen::ArrayXd>(&tmpy[0], tmpy.size());
// }

// //old version
// void  loadLib(const std::string& filename, Eigen::SparseMatrix<double,
// Eigen::RowMajor, std::ptrdiff_t>& X,Eigen::VectorXd& y ){
//   int j = 0, colLen = 0, maxIndex = 0;
//   std::vector<double> tmpy;

//   std::string buff;
//   std::ifstream ifs(filename);
//   std::list<std::string> results;
//   std::list<std::string> results1;
//   if (ifs.fail())
//   {
//     std::cerr << "can't open input file " << filename <<std::endl;
//     std::exit(1);
//   }
//   while (getline(ifs, buff))
//   {
//     ++colLen;
//     std::string::size_type lastColonIndex = buff.find_last_of(":");
//     if(lastColonIndex != std::string::npos ){
//       std::string tmpstring = buff.substr(0,lastColonIndex);
//       int lastElementIndex = string2int(tmpstring.substr(
//       tmpstring.find_last_of(" ") , tmpstring.length() -
//       tmpstring.find_last_of(" ")+1 ));
//       if(maxIndex < lastElementIndex) maxIndex = lastElementIndex;
//     }
//   }
//   X.resize(colLen, maxIndex);
//   ifs.close();
//   ifs.open(filename);
//   for(int i = 0; getline(ifs,buff); ++i){
//     // cout << buff <<endl;
//     boost::split(results, buff,  boost::is_space());
//     double yValue = string2double(results.front());
//     if(yValue == 0.0 ) yValue = -1.0;
//     tmpy.push_back(yValue);
//     results.pop_front();
//     j = 0;
//     BOOST_FOREACH(std::string p, results) {
//       if(p.find_first_of(":") != std::string::npos){
//         boost::split(results1, p,  boost::is_any_of(":"));
//         int tmpIndex = string2int(results1.front());
//         results1.pop_front();
//         X.insert(i,tmpIndex-1) = string2double(results1.front());
//         // v.push_back(T(i, tmpIndex, string2double(results1.front()) ));
//         ++j;
//       }
//     }
//   }
//   // X.setFromTriplets(v.begin(), v.end());
//   y = Eigen::Map<Eigen::VectorXd>(&tmpy[0], tmpy.size());
// }

// void load_model(const char *fname, Eigen::VectorXd& w) {
//   std::ifstream fs(fname);
//   if (fs.bad() || fs.fail()) {
//     std::cout << "file open error" << std::endl; std::exit(1);
//   }
//   std::vector<double> tmpy;

//   std::string buf;

//   while (std::getline(fs, buf))
//   {
//     if(buf.find_first_of(".") == std::string::npos) continue;
//     tmpy.push_back(string2double(buf));
//   }
//   w = Eigen::Map<Eigen::ArrayXd>(&tmpy[0], tmpy.size());
// }

// void loadLib(const char *fname, Eigen::SparseMatrix<double,
// Eigen::RowMajor>& X,Eigen::ArrayXd& y ){
//   std::ifstream fs(fname);
//   if (fs.bad() || fs.fail()) {
//     std::cout << "file open error" << std::endl; std::exit(1);
//   }
//   typedef Eigen::Triplet<double> T;
//   std::vector<T> tripletList;
//   tripletList.reserve(1024); // estimation of non_zero_entries
//   y.resize(1024);

//   std::string buf;
//   int n, d;
//   n = d = 0;

//   while (std::getline(fs, buf))
//   {
//     std::string::size_type idx1 = 0, idx2 = 0;

//     if ( (idx1 = buf.find_first_of(" \t", 0)) == std::string::npos ) {
//       std::cout << "file format error1" << std::endl; std::exit(1);
//     }

//     // use a C code "strtod/strtod", because of C++ "sstream" is slow.
//     // In addition, if you use strtod and std::string, you have to use
//     c_str() is too slow,
//     // so you should not use strtod together with std::string.
//     y[n] = strtod(buf.substr(0, idx1).c_str(), NULL);
//     y[n] = (y[n] > 0.0) ? 1.0 : -1.0;

//     // tripletList.push_back(T(n, 0, 1.0)); // bias

//     do {
//       --idx1;
//       ++idx1;
//       if ( (idx2 = buf.find_first_of(":", idx1)) == std::string::npos )
//         break;

//       int k = strtod(buf.substr(idx1, idx2 - idx1).c_str(), NULL);
//       if (d < k) d = k;

//       ++idx2;
//       idx1 = buf.find_first_of(" \t", idx2);

//       double tmp = strtod(buf.substr(idx2, idx1 - idx2).c_str(), NULL);
//       tripletList.push_back(T(n, k, tmp));
//     }
//     while (idx1 != std::string::npos);

//     if (y.size() <= (++n))
//       y.conservativeResize(y.size() * 2);
//   }
//   d += 1; // for bias
//   fs.close();

//   y.conservativeResize(n);
//   X.resize(n, d);
//   X.setFromTriplets(tripletList.begin(), tripletList.end());
//   X.makeCompressed();
// }

// void  loadLib(const std::string& filename, Eigen::SparseMatrix<double,
// Eigen::RowMajor>& X,Eigen::ArrayXd& y ){
//   int colLen = 0, maxIndex = 0, tmpIndex , nnz=0, num_posi=0, num_nega=0;
//   double yValue, yValueType1 = -1.0, yValueType2 = 1.0;
//   bool flag1 = false , flag2 = false;
//   std::vector<double> tmpy;
//   typedef Eigen::Triplet<double> T;
//   std::vector<T> v;

//   std::string buff;
//   std::ifstream ifs(filename);
//   std::list<std::string> results;
//   std::list<std::string> results1;
//   if (ifs.fail())
//   {
//     std::cerr << "can't open input file " << filename <<std::endl;
//     std::exit(1);
//   }

//   while (getline(ifs, buff))
//   {
//     boost::split(results, buff,  boost::is_space());
//     yValue = string2double(results.front());
//     if(!flag1){
//       yValueType1 = yValue;
//       flag1 = true;
//     }
//     else if(!flag2 && yValue != yValueType1){
//       yValueType2 = yValue;
//       flag2 = true;
//     }
//     if(flag1 && flag2) break;
//   }
//   ifs.close();
//   ifs.open(filename);

//   if(yValueType1 > yValueType2){
//     double tmp = yValueType2;
//     yValueType2 = yValueType1;
//     yValueType1 = tmp;
//   }
//   // std::cout << yValueType1 <<" " <<yValueType2 <<std::endl;
//   if(yValueType1 == -1.0 && yValueType2 == 1.0){
//     while (getline(ifs, buff))
//     {
//       boost::split(results, buff,  boost::is_space());
//       yValue = string2double(results.front());
//       tmpy.push_back(yValue);
//       results.pop_front();
//       if(yValue == -1.0) {
//         ++num_nega;
//       }
//       else {
//         ++num_posi;
//       }
//       BOOST_FOREACH(std::string p, results) {
//         tmpIndex = 0;
//         if(p.find_first_of(":") != std::string::npos){
//           ++nnz;
//           boost::split(results1, p,  boost::is_any_of(":"));
//           tmpIndex = string2int(results1.front());
//           results1.pop_front();
//           v.push_back(T(colLen, tmpIndex-1, string2double(results1.front())
//           ));
//         }
//         if(maxIndex < tmpIndex) maxIndex = tmpIndex;
//       }
//       ++colLen;
//     }
//   }
//   else{
//     while (getline(ifs, buff))
//     {
//       boost::split(results, buff,  boost::is_space());
//       yValue = string2double(results.front());
//       if(yValue == yValueType1) {
//         tmpy.push_back(-1.0);
//         ++num_nega;
//       }
//       else {
//         tmpy.push_back(1.0);
//         ++num_posi;
//       }
//       results.pop_front();

//       BOOST_FOREACH(std::string p, results) {
//         tmpIndex = 0;
//         if(p.find_first_of(":") != std::string::npos){
//           ++nnz;
//           boost::split(results1, p,  boost::is_any_of(":"));
//           tmpIndex = string2int(results1.front());
//           results1.pop_front();
//           v.push_back(T(colLen, tmpIndex-1, string2double(results1.front())
//           ));
//         }
//         if(maxIndex < tmpIndex) maxIndex = tmpIndex;
//       }
//       ++colLen;
//     }
//   }
//   std::cout <<"#data"<< colLen <<" #Posi " <<num_posi <<" #Nega " <<num_nega
//   <<" #features" <<maxIndex <<" #nz " << nnz <<std::endl;
//   X.resize(colLen, maxIndex);

//   X.setFromTriplets(v.begin(), v.end());
//   y = Eigen::Map<Eigen::ArrayXd>(&tmpy[0], tmpy.size());
// }
