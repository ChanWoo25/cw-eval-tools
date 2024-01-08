// My Point Cloud Tools
#ifndef CSVROW_HPP_
#define CSVROW_HPP_

#include <algorithm>
#include <string>
#include <vector>
#include <fstream>

namespace cxxutil {

struct isSpace
{
  bool operator()(unsigned c)
  {
    return (c == ' ' || c == '\n' || c == '\r' ||
            c == '\t' || c == '\v' || c == '\f');
  }
};

class CSVRow
{
private:
  std::string         line_;
  std::vector<int>    data_;

public:
  inline std::size_t size() const { return data_.size() - 1; }

  std::string_view operator[](std::size_t index) const
  {
    return std::string_view(&line_[data_[index] + 1], data_[index + 1] -  (data_[index] + 1));
  }

  void readNextRow(std::istream & str)
  {
    do {
      std::getline(str, line_);
    } while (line_[0] == '#');

    data_.clear();
    data_.emplace_back(-1);
    std::string::size_type pos = 0;

    line_.erase(std::remove_if(line_.begin(), line_.end(), isSpace()), line_.end());

    while((pos = line_.find(',', pos)) != std::string::npos)
    {
      data_.emplace_back(pos);
      ++pos;
    }

    // This checks for a trailing comma with no data after it.
    pos   = line_.size();
    data_.emplace_back(pos);
  }
};

std::istream & operator >> (
  std::istream & str,
  CSVRow & data);

}; // namespace cxxutil

#endif
