//
// Author: Chanwoo Lee (https://github.com/ChanWoo25)
//

#include <spdlog/common.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <archive.hpp>

/**
https://leetcode.com/discuss/interview-experience/3171859/Journey-to-a-FAANG-Company-Amazon-or-SDE2-(L5)-or-Bangalore-or-Oct-2022-Accepted

Summary

 every day check the mail in the morning and second to keep on applying for open positions on daily basis.
He was confident on wrting code, but not on the algorithms, so he just watched videos only.
=> Unfortunately He was not able to write any code in first two coding rounds.

Things to remember for DSA round :
1. Discuss the problem before deep diving into it.
2. Start with a brute force solution without thinking much on optimisation.
3. Be vocal while writing code and verifying your code - keep telling whatever you are thinking.
4. Must dry run your code and effort spent on thinking about corner cases.
5. If asked with behavioural questions, then keep it short and direct - to save more time for coding.

Things to do for DSA preparation:
1. Complete all the topic wise sections in the study plan section of the leet code.
2. Practice questions in a time constrained manner, no point of spending hours in a single question.
3. Easy and medium level questions are sufficient enough to crack DSA rounds.
4. Once you complete the study plan you can refer to the tutorial by Fraz on a set of 250 questions.
   One thing I would like to suggest is to remove the topic header in the excel sheet, so that
   when you attempt a question from the sheet you should not be aware of which topic it is from and
   that you should identify yourself. Also keep nothing down things you learnt from each of the
   questions - this will help you to revise concepts later. Repeat good questions/ questions that
   you find difficult to solve and keep repeating until you are convinced.
5. Few of the tutorials that I followed :
  DP, Binary Search & Stack : https://www.youtube.com/@TheAdityaVerma
  Graph, Tree, Trie : https://m.youtube.com/@takeUforward
  C++ Tutorial: https://m.youtube.com/playlist?list=PLauivoElc3ggagradg8MfOZreCMmXMmJ-
  Basic DSA Concepts : Jennys’ Classes & Abdul Baris’s Tutorial
  Two of the good coding books were added to the github
  link given below.

*/
#include <fstream>

int main()
{
  int a = 4, b = 3, c = 3, d = 4, e = 5, f = 2;
  auto output00 = Solution::p10036_minMovesToCaptureTheQueen(a, b, c, d, e, f);
  spdlog::info("Answer00: {}", output00);

  return EXIT_SUCCESS;
}
