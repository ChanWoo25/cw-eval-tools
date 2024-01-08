#include <archive.hpp>

int Solution::p10035_areaOfMaxDiagonal(
  std::vector<std::vector<int>>& dimensions)
{
  double max_diag2 = std::numeric_limits<double>::min();
  double wh_diff = -1.0;
  double area = -1.0;
  for (const auto & wh: dimensions)
  {
      const auto w = static_cast<double>(wh[0]);
      const auto h = static_cast<double>(wh[1]);

      const auto diag2 = std::pow(w,2) + std::pow(h,2);
      if (max_diag2 < diag2)
      {
        max_diag2 = diag2;
        wh_diff = std::abs(w - h);
        area = w * h;
      }
      else if (std::abs(max_diag2 - diag2) < DBL_EPS)
      {
        if (wh_diff > std::abs(w - h))
        {
          wh_diff = std::abs(w - h);
          area = w * h;
        }
      }
  }

  return area;
}

  // auto input00 = VVI();
  // input00.push_back({9, 3});
  // input00.push_back({8, 6});

  // auto output00 = Solution::p10035_areaOfMaxDiagonal(input00);
  // spdlog::info("Answer00: {}", output00);

  // auto input01 = VVI();
  // input01.push_back({3, 4});
  // input01.push_back({4, 3});

  // auto output01 = Solution::p10035_areaOfMaxDiagonal(input01);
  // spdlog::info("Answer01: {}", output01);

/**
 * @brief on 8x8 chessboard with given rook, bishop, queen coord
 * 3번을 움직여야 하는 경우는 없다.
 * bishop은 rook 2번을 대체할 수도있지만, 절반밖에 커버하지 못한다.
 * 1. rook으로 바로 잡을 수 있는 경우. => 1
 * 2. rook과 직선상, 사이에 비숍이 있는 경우, => 2
 * 3. Otherwise, bishop과의 직선거리 상에 rook이 있지 않은 이상,
                 모든 경우에 rook이 2번 안에 잡을 수 있음.
 *
 복기
 - 조건문이 많은 것은 둘째치고, 일관성이 부족하다.
 - 1번 혹은 2번만이 답이 될 수 있음을 사전에 알고 있다면, 반복문 따위보다는
   조건문으로 푸는 것이 좋다. 또한 경우의 수가 적은 1번 Case를 앞으로 빼고,
   2번은 Else 문으로 넘기려고 목표를 정해놓고 구현했으면 좀 더 Clear했을 것이다.
 */
int Solution::p10036_minMovesToCaptureTheQueen(
  int a, int b,
  int c, int d,
  int e, int f)
{
  int rx = a, ry = b; // rook
  int bx = c, by = d; // bishop
  int qx = e, qy = f; // queen
  spdlog::info("rook ({}, {})", rx, ry);
  spdlog::info("bishop ({}, {})", bx, by);
  spdlog::info("queen ({}, {})", qx, qy);
  if ((rx == qx && rx != bx) || // Case 1
      (ry == qy && ry != by))
  {
    return 1;
  }
  else if (rx == qx && rx == bx) // Case 2
  {
    if ((by < ry && by < qy) ||
        (by > ry && by > qy))
    {
      return 1;
    }
  }
  else if (ry == qy && ry == by) // Case 2
  {
    if ((bx < rx && bx < qx) ||
        (bx > rx && bx > qx))
    {
      return 1;
    }
  }
  else
  {
    if ((qx + qy) % 2 != (bx + by) % 2) // Never
    {
      return 2;
    }
    if (std::abs(bx - qx) != std::abs(by - qy)) // Same line
    {
      return 2;
    }
    else
    {
      int xstep = (qx - bx) / std::abs(bx - qx);
      int ystep = (qy - by) / std::abs(by - qy);
      spdlog::info("xstep={}, ystep={}", xstep, ystep);
      for (int x = bx, y = by; x != qx; x += xstep, y += ystep)
      {
        if (x == rx && y == ry)
        {
          return 2;
        }
      }
      return 1;
    }
    return 2;
  }

  return 2;
}
