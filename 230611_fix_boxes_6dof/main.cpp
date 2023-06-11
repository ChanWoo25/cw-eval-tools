#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdio>

int main (int argc, char * argv[])
{
  const double TIME_OFFSET = 1468941032.22916555;

  {
    std::ifstream ifile("/data/datasets/dataset_ecd/boxes_6dof/events.txt");
    std::ofstream ofile("/home/leecw/boxes_6dof_events.txt");
    if (ifile.is_open()) { std::cout << "ifile success!" << std::endl; }
    if (ofile.is_open()) { std::cout << "ofile success!" << std::endl; }
    double t; int x, y, p;
    while (!ifile.eof())
    {
      ifile >> t >> x >> y >> p;
      t -= TIME_OFFSET;
      char buffer[100];
      sprintf(buffer, "%.9f %d %d %d\n", t, x, y, p);
      ofile << buffer;
    }
    ifile.close();
    ofile.close();
  }

  { // GT
    std::ifstream gt_ifile("/data/datasets/dataset_ecd/boxes_6dof/groundtruth.txt");
    std::ofstream gt_ofile("/home/leecw/boxes_6dof_groundtruth.txt");
    if (gt_ifile.is_open()) { std::cout << "gt_ifile success!" << std::endl; }
    if (gt_ofile.is_open()) { std::cout << "gt_ofile success!" << std::endl; }
    double t, _x, _y, _z, _qx, _qy, _qz, _qw;
    while (!gt_ifile.eof())
    {
      gt_ifile >> t >> _x >> _y >> _z >> _qx >> _qy >> _qz >> _qw;
      char buffer[200];
      t -= TIME_OFFSET;
      sprintf(buffer, "%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
              t, _x, _y, _z, _qx, _qy, _qz, _qw);
      gt_ofile << buffer;
      // std::cout << std::string(buffer);
    }
    gt_ifile.close();
    gt_ofile.close();
  }

  { // imu
    std::ifstream ifile("/data/datasets/dataset_ecd/boxes_6dof/imu.txt");
    std::ofstream ofile("/home/leecw/boxes_6dof_imu.txt");
    if (ifile.is_open()) { std::cout << "ifile success!" << std::endl; }
    if (ofile.is_open()) { std::cout << "ofile success!" << std::endl; }
    double t, ax, ay, az, gx, gy, gz;
    while (!ifile.eof())
    {
      ifile >> t >> ax >> ay >> az >> gx >> gy >> gz;
      char buffer[200];
      t -= TIME_OFFSET;
      sprintf(buffer, "%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
              t, ax, ay, az, gx, gy, gz);
      ofile << buffer;
      // std::cout << std::string(buffer);
    }
    ifile.close();
    ofile.close();
  }

  return EXIT_SUCCESS;
}
