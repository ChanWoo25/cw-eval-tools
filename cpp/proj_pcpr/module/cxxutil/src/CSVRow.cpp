#include <cxxutil/CSVRow.hpp>

namespace cxxutil {

std::istream & operator >> (
  std::istream & str,
  CSVRow & data)
{
  data.readNextRow(str);
  return str;
}

}; // namespace cxxutil

