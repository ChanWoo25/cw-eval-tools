#! /bin/bash
cmake -B build -S .
cmake --build build --target fix_boxes_6dof
./bin/fix_boxes_6dof
